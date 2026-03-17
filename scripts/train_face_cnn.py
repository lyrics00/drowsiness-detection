"""
Train FaceDrowsinessCNN (Pretrained ResNet18) on face crops.

Expected structure:
data/processed/Face/
    train/Active, train/Fatigue
    val/Active,   val/Fatigue
    test/Active,  test/Fatigue

Run:
    python scripts/train_face_cnn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import os

# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = 224

DATA_DIR = Path("data/processed/Face")
MODEL_SAVE_PATH = Path("outputs/runs/face_cnn_model.pth")


# -------------------------
# Model (Pretrained ResNet18)
# -------------------------
class FaceDrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Data loaders
# -------------------------
def make_loaders(data_dir: Path, batch_size: int):

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],   # ImageNet mean
            [0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=eval_tf)

    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# -------------------------
# Main Training
# -------------------------
def main():

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_loader, val_loader, test_loader = make_loaders(
        DATA_DIR, BATCH_SIZE
    )

    model = FaceDrowsinessCNN().to(device)

    # Freeze backbone initially
    for param in model.model.parameters():
        param.requires_grad = False

    # Unfreeze last block + classifier
    for param in model.model.layer4.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # -------- Train --------
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # -------- Evaluate --------
        train_loss, train_acc = evaluate(
            model, train_loader, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch:2d}/{EPOCHS}] "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  ✔ Saved best model")

        # After 8 epochs → unfreeze full model
        if epoch == 8:
            print("Unfreezing entire backbone for fine-tuning...")
            for param in model.model.parameters():
                param.requires_grad = True

    # -------- Test --------
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print("\nFinal Results:")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()