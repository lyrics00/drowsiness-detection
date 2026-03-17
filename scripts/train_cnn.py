import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 64

# paths
DATA_DIR = os.path.join("data", "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "Train_Eye")
VAL_DIR = os.path.join(DATA_DIR, "Validation_Eye")
MODEL_SAVE_PATH = os.path.join("outputs", "runs", "eye_cnn_model.pth")

# neurl network architecture
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
            nn.Linear(128, 2) # 2 classes: Closed (0), Opened (1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # load data
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes: {train_dataset.classes}")

    # initialize model, loss, and optimizer
    model = SimpleEyeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # val 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

    # save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()