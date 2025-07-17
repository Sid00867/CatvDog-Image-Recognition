from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # Input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32 x 128 x 128
        self.pool1 = nn.MaxPool2d(2, 2)                           # 32 x 64 x 64

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 x 64 x 64
        self.pool2 = nn.MaxPool2d(2, 2)                           # 64 x 32 x 32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128 x 32 x 32
        self.pool3 = nn.MaxPool2d(2, 2)                           # 128 x 16 x 16

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # 128 x 16 x 16
        self.pool4 = nn.MaxPool2d(2, 2)                            # 128 x 8 x 8

        self.flatten_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    # data loading
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    train_dir = os.path.join(base_dir, './training_set/training_set')
    test_dir = os.path.join(base_dir, './test_set/test_set')

    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create datasets with ImageFolder
    # It automatically finds classes from folder names (e.g., 'cats', 'dogs')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set up the optimizer and loss function
    model = CatDogCNN()
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("Started Training...")
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.float().unsqueeze(1) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    print("Training Ended.\nStarting Testing...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            targets = targets.float().unsqueeze(1) 
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_test_loss = test_loss / total
    accuracy = correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")