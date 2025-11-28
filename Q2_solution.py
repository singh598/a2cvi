import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ======================
# CUSTOM MNIST CSV DATASET
# ======================

class MNISTDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values.astype(np.float32) / 255.0
        self.images = self.images.reshape(-1, 1, 28, 28)  # convert to image shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.images[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ======================
# CNN MODEL
# ======================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)

# ======================
# TRAINING CODE
# ======================

def train_model(csv_path="mnist_train.csv", batch_size=64, epochs=10):
    dataset = MNISTDataset(csv_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Started...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss={running_loss:.4f}, Accuracy={accuracy:.2f}%")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Training done! Model saved as mnist_cnn.pth")

# ======================
# EVALUATION
# ======================

def evaluate(csv_path="mnist_test.csv"):
    dataset = MNISTDataset(csv_path)
    loader = DataLoader(dataset, batch_size=64)

    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nFinal Test Accuracy = {100 * correct / total:.2f}%")

# ======================
# MAIN

if __name__ == "__main__":
    train_model("mnist_train.csv", epochs=10)
    evaluate("mnist_test.csv")
