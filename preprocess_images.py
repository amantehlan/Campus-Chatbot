"""
Image preprocessing pipeline for the Campus Chatbot project.
- Resize and normalize images
- Apply augmentation (random crop, rotation, color jitter)
- Convert to tensors and build batch loaders
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

KB_PATH = "data/knowledge_base/locations.json"

# Standard ImageNet normalization stats (compatible with CLIP/ResNet/EfficientNet backbones)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class CampusImageDataset(Dataset):
    """Loads campus building images with their location label (id and category)."""

    def __init__(self, kb_path=KB_PATH, transform=None):
        with open(kb_path, "r") as f:
            self.kb = json.load(f)

        self.transform = transform
        self.samples = []
        self.label_to_idx = {record["id"]: i for i, record in enumerate(self.kb)}

        for record in self.kb:
            folder = record["image_folder"]
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({
                        "path": os.path.join(folder, fname),
                        "label_id": record["id"],
                        "label_idx": self.label_to_idx[record["id"]],
                        "name": record["name"],
                        "category": record["category"],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label_idx": sample["label_idx"],
            "label_id": sample["label_id"],
            "name": sample["name"],
        }


def get_dataloaders(batch_size=4, val_split=0.2, seed=42):
    """Returns train and validation DataLoaders with an 80/20 split."""
    full_dataset = CampusImageDataset(transform=train_transform)
    n = len(full_dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)

    # Apply eval transform to validation set
    val_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=2)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print("Sample batch:")
    print(f"  Image tensor shape: {batch['image'].shape}")
    print(f"  Labels: {batch['label_id']}")
    print(f"  Names: {batch['name']}")