import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 64

# paths
DATA_DIR = os.path.join("data", "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "Train_Eye")
VAL_DIR = os.path.join(DATA_DIR, "Validation_Eye")
MODEL_SAVE_PATH = os.path.join("outputs", "runs", "eye_cnn_model.pth")


# neural network architecture
class SimpleEyeCNN(nn.Module):
    def __init__(self):
        super(SimpleEyeCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes: Closed (0), Opened (1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- augmented train transform to close the gap to real webcam frames ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # val/test: no augmentation, just resize + normalise
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=VAL_DIR,   transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ImageFolder sorts alphabetically -> 0 = Closed, 1 = Opened
    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    model     = SimpleEyeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # halve LR when val accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # ---- validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"Train Loss: {running_loss/len(train_loader):.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f}  Acc: {val_acc:.2f}%"
            + (" <- best" if val_acc > best_val_acc else "")
        )

        # save only the best checkpoint (not just the last epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()