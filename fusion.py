"""
Multimodal fusion module for the Campus Chatbot project.

Design:
- Each modality (image, text/voice) produces an independent "vote" for a KB location:
    - Vision: CLIP+FAISS retrieval -> top-1 location id + confidence score
    - Text/Voice: DistilBERT intent classification + entity extraction -> location id (from KB lookup)
- A lightweight fusion MLP takes the concatenation of:
    - CLIP image embedding (512-dim) or a learned zero-vector if absent
    - DistilBERT pooled text embedding (768-dim) or a learned zero-vector if absent
    - presence flags (2-dim, indicating which modalities are active)
  and outputs a probability distribution over the 15 KB location ids.

Routing strategy:
- Single modality: that modality's embedding dominates; the zero-vector for the
  missing modality contributes (close to) nothing, since the MLP learns to ignore
  zero inputs when the presence flag is 0.
- Multiple modalities: both embeddings are concatenated and the MLP learns to weigh
  them jointly. This lets an image + a typed clarification ("the cafeteria one")
  jointly disambiguate between visually similar locations.

Trade-offs discussed in the report:
- A simple MLP with concatenated embeddings is easy to train but doesn't model
  cross-modal attention; a transformer-based fusion layer would be more expressive
  but requires far more data than this project's 15-location, ~50-image dataset.
- Padding absent modalities with zero vectors plus an explicit presence flag is a
  cheap way to let a single small MLP handle both single- and multi-modality cases
  without needing separate models per input combination.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KB_PATH = "data/knowledge_base/locations.json"
FUSION_MODEL_DIR = "models/fusion"

CLIP_DIM = 512
TEXT_DIM = 768  # DistilBERT pooled output dim


class FusionMLP(nn.Module):
    """Lightweight MLP that maps concatenated modality embeddings to a KB location id."""

    def __init__(self, clip_dim=CLIP_DIM, text_dim=TEXT_DIM, num_classes=15, hidden_dim=256):
        super().__init__()
        input_dim = clip_dim + text_dim + 2  # +2 for presence flags
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, image_emb, text_emb, image_present, text_present):
        """
        image_emb: (B, clip_dim) - zeros if image absent
        text_emb: (B, text_dim) - zeros if text absent
        image_present: (B, 1) - 1.0 if image provided, else 0.0
        text_present: (B, 1) - 1.0 if text provided, else 0.0
        """
        x = torch.cat([image_emb, text_emb, image_present, text_present], dim=-1)
        return self.net(x)


class FusionRouter:
    """
    Inference-time router. Wraps the FusionMLP plus rule-based fallbacks so the
    system works even before the MLP is trained (using CLIP/DistilBERT signals
    directly), and can fall back gracefully when only one modality is present.
    """

    def __init__(self, kb_path=KB_PATH, model_path=None):
        with open(kb_path, "r") as f:
            self.kb = json.load(f)
        self.id_to_record = {r["id"]: r for r in self.kb}
        self.id_list = [r["id"] for r in self.kb]
        self.label_to_idx = {loc_id: i for i, loc_id in enumerate(self.id_list)}
        self.idx_to_label = {i: loc_id for loc_id, i in self.label_to_idx.items()}

        self.mlp = FusionMLP(num_classes=len(self.id_list))
        self.mlp.eval()

        if model_path and os.path.exists(model_path):
            self.mlp.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded fusion MLP weights from {model_path}")
        else:
            print("No trained fusion MLP found - using rule-based routing fallback.")

    def route(self, image_result=None, text_result=None, image_emb=None, text_emb=None):
        """
        image_result: dict from CLIP+FAISS retrieval, e.g.
            {"id": "main_library", "score": 0.31, ...} or None
        text_result: dict from intent/entity extraction, e.g.
            {"intent": "find_location", "entity": "main_library"} or None
        image_emb: optional (512,) numpy array, CLIP image embedding
        text_emb: optional (768,) numpy array, DistilBERT pooled embedding

        Returns: matched KB record + routing metadata
        """
        # --- Rule-based fast path (works without trained fusion MLP) ---

        # Case 1: only text/voice input, entity recognized directly
        if text_result and text_result.get("entity") and not image_result:
            loc_id = text_result["entity"]
            return self._build_response(loc_id, source="text_entity", confidence=1.0)

        # Case 2: only image input
        if image_result and not (text_result and text_result.get("entity")):
            loc_id = image_result["id"]
            return self._build_response(loc_id, source="image_retrieval", confidence=image_result.get("score", 0.0))

        # Case 3: both modalities present and agree -> high confidence
        if image_result and text_result and text_result.get("entity"):
            if image_result["id"] == text_result["entity"]:
                return self._build_response(
                    image_result["id"], source="image+text_agreement", confidence=1.0
                )
            # Disagreement: if a trained fusion MLP is available, use it; otherwise
            # prefer the text entity (typed/spoken intent is usually more explicit
            # about *which* location, while image retrieval may be confused by
            # visually similar buildings).
            if image_emb is not None and text_emb is not None:
                return self._fusion_mlp_predict(image_emb, text_emb)
            return self._build_response(
                text_result["entity"], source="text_priority_on_disagreement", confidence=0.6,
                note=f"Image suggested '{image_result['id']}' but text entity took priority."
            )

        # Case 4: nothing usable
        return {
            "matched": False,
            "message": "Could not determine a location from the given input.",
            "source": "none",
        }

    def _fusion_mlp_predict(self, image_emb, text_emb):
        with torch.no_grad():
            img_t = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0)
            txt_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0)
            img_present = torch.ones((1, 1))
            txt_present = torch.ones((1, 1))
            logits = self.mlp(img_t, txt_t, img_present, txt_present)
            probs = F.softmax(logits, dim=-1)
            top_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, top_idx].item())

        loc_id = self.idx_to_label[top_idx]
        return self._build_response(loc_id, source="fusion_mlp", confidence=confidence)

    def _build_response(self, loc_id, source, confidence, note=None):
        record = self.id_to_record.get(loc_id)
        if record is None:
            return {"matched": False, "message": f"Unknown location id '{loc_id}'", "source": source}

        response = {
            "matched": True,
            "source": source,
            "confidence": round(confidence, 3),
            "location": {
                "id": record["id"],
                "name": record["name"],
                "category": record["category"],
                "description": record["description"],
                "opening_hours": record["opening_hours"],
                "map_reference": record["location"]["map_reference"],
                "events": record.get("events", []),
            },
        }
        if note:
            response["note"] = note
        return response


def train_fusion_mlp_synthetic(num_classes=15, epochs=30, save_path=os.path.join(FUSION_MODEL_DIR, "fusion_mlp.pt")):
    """
    Trains the fusion MLP on synthetic single-modality examples derived from the KB,
    so that even with no labelled multimodal pairs, the MLP learns a sensible
    identity mapping: a clean CLIP/text embedding for location X should map to class X.

    In a full project, this would be replaced with real (image_emb, text_emb, label)
    triples collected from your dataset and voice/text corpus.
    """
    os.makedirs(FUSION_MODEL_DIR, exist_ok=True)

    with open(KB_PATH, "r") as f:
        kb = json.load(f)

    num_classes = len(kb)
    model = FusionMLP(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    losses = []
    for epoch in range(epochs):
        # Synthetic batch: for each class, generate a random embedding "centroid"
        # plus noise, simulating real CLIP/DistilBERT embeddings for that location.
        batch_img = []
        batch_txt = []
        batch_labels = []
        batch_img_present = []
        batch_txt_present = []

        torch.manual_seed(epoch)
        for class_idx in range(num_classes):
            centroid_img = torch.randn(CLIP_DIM) * 0.1 + class_idx
            centroid_txt = torch.randn(TEXT_DIM) * 0.1 + class_idx

            # Image-only sample
            batch_img.append(centroid_img + torch.randn(CLIP_DIM) * 0.05)
            batch_txt.append(torch.zeros(TEXT_DIM))
            batch_img_present.append(1.0)
            batch_txt_present.append(0.0)
            batch_labels.append(class_idx)

            # Text-only sample
            batch_img.append(torch.zeros(CLIP_DIM))
            batch_txt.append(centroid_txt + torch.randn(TEXT_DIM) * 0.05)
            batch_img_present.append(0.0)
            batch_txt_present.append(1.0)
            batch_labels.append(class_idx)

            # Both modalities present
            batch_img.append(centroid_img + torch.randn(CLIP_DIM) * 0.05)
            batch_txt.append(centroid_txt + torch.randn(TEXT_DIM) * 0.05)
            batch_img_present.append(1.0)
            batch_txt_present.append(1.0)
            batch_labels.append(class_idx)

        img_t = torch.stack(batch_img)
        txt_t = torch.stack(batch_txt)
        img_present_t = torch.tensor(batch_img_present).unsqueeze(-1)
        txt_present_t = torch.tensor(batch_txt_present).unsqueeze(-1)
        labels_t = torch.tensor(batch_labels)

        logits = model(img_t, txt_t, img_present_t, txt_present_t)
        loss = F.cross_entropy(logits, labels_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved fusion MLP to {save_path}")
    return losses


if __name__ == "__main__":
    print("=== Training fusion MLP on synthetic single/multi-modality embeddings ===")
    train_fusion_mlp_synthetic()

    print("\n=== Testing FusionRouter ===")
    router = FusionRouter(model_path=os.path.join(FUSION_MODEL_DIR, "fusion_mlp.pt"))

    # Scenario 1: text only (e.g., transcribed voice query "Where is the main library?")
    result = router.route(text_result={"intent": "find_location", "entity": "main_library"})
    print("\nScenario 1 (text only):")
    print(json.dumps(result, indent=2))

    # Scenario 2: image only (e.g., CLIP+FAISS top-1)
    result = router.route(image_result={"id": "fat_cat_cafe", "score": 0.28})
    print("\nScenario 2 (image only):")
    print(json.dumps(result, indent=2))

    # Scenario 3: image + text agree
    result = router.route(
        image_result={"id": "bedlam_theatre", "score": 0.30},
        text_result={"intent": "find_event", "entity": "bedlam_theatre"},
    )
    print("\nScenario 3 (image + text agree):")
    print(json.dumps(result, indent=2))

    # Scenario 4: image + text disagree, no embeddings (rule-based fallback)
    result = router.route(
        image_result={"id": "george_square_55_60", "score": 0.22},
        text_result={"intent": "find_location", "entity": "main_library"},
    )
    print("\nScenario 4 (image + text disagree):")
    print(json.dumps(result, indent=2))

    # Scenario 5: nothing
    result = router.route()
    print("\nScenario 5 (no input):")
    print(json.dumps(result, indent=2))