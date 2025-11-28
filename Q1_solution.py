import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. DATA TRANSFORMS
# ===============================

train_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
])

# ===============================
# 2. LOAD DATA
# ===============================

train_dataset = datasets.ImageFolder("train", transform=train_transform)
test_dataset = datasets.ImageFolder("test", transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# ===============================
# 3. SIMPLE CNN MODEL
# ===============================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*37*37, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 4. TRAINING
# ===============================

epochs = 15
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{epochs}, Loss = {running_loss/len(train_loader):.4f}")

# ===============================
# 5. SAVE MODEL
# ===============================

torch.save(model.state_dict(), "cat_dog_model.pth")
print("Model saved as cat_dog_model.pth")

# ===============================
# 6. TESTING ON TEST FOLDER IMAGES
# ===============================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        output = model(imgs)
        _, predicted = torch.max(output, 1)
        total += 1
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

# ===============================
# 7. TESTING WITH INTERNET IMAGES
# ===============================

from PIL import Image

def predict_image(path):
    img = Image.open(path)
    img_t = test_transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    print(f"{path} → Predicted: {class_names[pred.item()]}")

# Example usage:
predict_image("test_cat_from_web.jpg")
predict_image("test_dog_from_web.jpg")
