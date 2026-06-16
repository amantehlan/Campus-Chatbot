"""
Data exploration script for the Campus Chatbot project.
- Loads knowledge base
- Displays sample images per location with annotations
- Plots class distribution (images per location/category)
"""

import json
import os
import matplotlib.pyplot as plt
from PIL import Image

KB_PATH = "data/knowledge_base/locations.json"
IMAGES_ROOT = "data/images"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_knowledge_base(path=KB_PATH):
    with open(path, "r") as f:
        return json.load(f)


def count_images_per_location(kb):
    counts = {}
    for record in kb:
        folder = record["image_folder"]
        if os.path.isdir(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            counts[record["name"]] = len(images)
        else:
            counts[record["name"]] = 0
    return counts


def plot_class_distribution(counts, save_path=os.path.join(OUTPUT_DIR, "class_distribution.png")):
    names = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color="steelblue")
    plt.xlabel("Number of Images")
    plt.title("Image Count per Campus Location")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to {save_path}")


def plot_category_distribution(kb, save_path=os.path.join(OUTPUT_DIR, "category_distribution.png")):
    cat_counts = {}
    for record in kb:
        cat = record["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    plt.figure(figsize=(8, 5))
    plt.bar(cat_counts.keys(), cat_counts.values(), color="darkorange")
    plt.ylabel("Number of Locations")
    plt.title("Locations per Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved category distribution plot to {save_path}")


def display_sample_images(kb, n_per_location=1, save_path=os.path.join(OUTPUT_DIR, "sample_images.png")):
    """Create a grid of sample images annotated with location name and category."""
    samples = []
    for record in kb:
        folder = record["image_folder"]
        if not os.path.isdir(folder):
            continue
        images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for img_name in images[:n_per_location]:
            samples.append((record["name"], record["category"], os.path.join(folder, img_name)))

    if not samples:
        print("No images found. Check that images are placed in data/images/<folder>/")
        return

    n = len(samples)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, (name, category, img_path) in zip(axes, samples):
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Image error", ha="center", va="center")
        ax.set_title(f"{name}\n({category})", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved annotated sample image grid to {save_path}")


if __name__ == "__main__":
    kb = load_knowledge_base()
    print(f"Loaded {len(kb)} location records.")

    counts = count_images_per_location(kb)
    print("Images per location:")
    for name, c in counts.items():
        print(f"  {name}: {c}")

    plot_class_distribution(counts)
    plot_category_distribution(kb)
    display_sample_images(kb, n_per_location=1)