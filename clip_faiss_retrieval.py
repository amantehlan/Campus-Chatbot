"""
CLIP + FAISS retrieval pipeline for the Campus Chatbot project.

Approach:
- Encode each knowledge base location's text description using CLIP's text encoder.
- Build a FAISS index over these text embeddings.
- At inference time, encode an uploaded image with CLIP's image encoder.
- Retrieve the nearest text embedding(s) via cosine similarity -> matched location(s).

This is a zero-shot retrieval approach: no labelled image training data required.
"""

import os
import json
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

KB_PATH = "data/knowledge_base/locations.json"
INDEX_DIR = "models/clip_faiss"
MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    print(f"Loading CLIP model '{MODEL_NAME}' on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def build_text_descriptions(kb):
    """Build a rich text description per location for CLIP text encoding."""
    descriptions = []
    for record in kb:
        desc = f"{record['name']}, a {record['category'].replace('_', ' ')}. {record['description']}"
        descriptions.append(desc)
    return descriptions


def build_index(model, processor, kb_path=KB_PATH, index_dir=INDEX_DIR):
    """Encode KB text descriptions, build and save a FAISS index."""
    with open(kb_path, "r") as f:
        kb = json.load(f)

    descriptions = build_text_descriptions(kb)

    with torch.no_grad():
        inputs = processor(text=descriptions, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output  # shape: (N, hidden_dim)
        text_features = model.text_projection(pooled_output)  # shape: (N, projection_dim)

        print(f"DEBUG text_features shape: {text_features.shape}")
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy().astype("float32")

    # Cosine similarity via inner product on normalized vectors
    dim = text_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(text_features)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "locations.index"))

    # Save id mapping
    id_map = [record["id"] for record in kb]
    with open(os.path.join(index_dir, "id_map.json"), "w") as f:
        json.dump(id_map, f, indent=2)

    print(f"Built FAISS index with {index.ntotal} entries -> saved to {index_dir}")
    return index, id_map, kb


def load_index(index_dir=INDEX_DIR, kb_path=KB_PATH):
    index = faiss.read_index(os.path.join(index_dir, "locations.index"))
    with open(os.path.join(index_dir, "id_map.json"), "r") as f:
        id_map = json.load(f)
    with open(kb_path, "r") as f:
        kb = json.load(f)
    return index, id_map, kb


def encode_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output  # shape: (1, hidden_dim)
        image_features = model.visual_projection(pooled_output)  # shape: (1, projection_dim)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype("float32")


def retrieve_location(model, processor, index, id_map, kb, image_path, top_k=3):
    """Return top_k matched location records for an uploaded image, with similarity scores."""
    query_vec = encode_image(model, processor, image_path)
    scores, indices = index.search(query_vec, top_k)

    kb_by_id = {record["id"]: record for record in kb}
    results = []
    for score, idx in zip(scores[0], indices[0]):
        loc_id = id_map[idx]
        record = kb_by_id[loc_id]
        results.append({
            "id": loc_id,
            "name": record["name"],
            "category": record["category"],
            "score": float(score),
        })
    return results


def evaluate_retrieval(model, processor, index, id_map, kb):
    """Evaluate top-1 and top-3 retrieval accuracy using held-out images per location."""
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    per_location_results = []

    for record in kb:
        folder = record["image_folder"]
        if not os.path.isdir(folder):
            continue
        images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            results = retrieve_location(model, processor, index, id_map, kb, img_path, top_k=3)
            predicted_ids = [r["id"] for r in results]

            is_top1 = predicted_ids[0] == record["id"]
            is_top3 = record["id"] in predicted_ids

            correct_top1 += int(is_top1)
            correct_top3 += int(is_top3)
            total += 1

            per_location_results.append({
                "image": img_path,
                "true_label": record["id"],
                "predicted_top1": predicted_ids[0],
                "predicted_top3": predicted_ids,
                "top1_correct": is_top1,
                "top3_correct": is_top3,
            })

    top1_acc = correct_top1 / total if total else 0
    top3_acc = correct_top3 / total if total else 0

    print(f"\n=== CLIP + FAISS Retrieval Evaluation ===")
    print(f"Total images evaluated: {total}")
    print(f"Top-1 accuracy: {top1_acc:.3f}")
    print(f"Top-3 accuracy: {top3_acc:.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/clip_retrieval_results.json", "w") as f:
        json.dump({
            "top1_accuracy": round(top1_acc, 3),
            "top3_accuracy": round(top3_acc, 3),
            "total_images": total,
            "details": per_location_results,
        }, f, indent=2)
    print("Saved results to outputs/clip_retrieval_results.json")

    return top1_acc, top3_acc


if __name__ == "__main__":
    model, processor = load_clip()
    index, id_map, kb = build_index(model, processor)
    evaluate_retrieval(model, processor, index, id_map, kb)