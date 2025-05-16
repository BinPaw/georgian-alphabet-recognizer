import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from georgian_handwritten_dataset import GeorgianHandwrittenDataset
from georgian_common import GEORGIAN_ALPHABET, SimpleCNN, DATA_DIR

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Ensure models directory exists
MODELS_DIR = 'models\\003'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Dataset
dataset = GeorgianHandwrittenDataset(root_dir=DATA_DIR)
num_classes = len(dataset.class_to_idx)
print(f"Number of classes: {num_classes}")
print(f"Total samples: {len(dataset)}")

# Stratified train/val split
labels = [label for _, label in dataset.samples]
labels = np.array(labels)
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
for train_idx, val_idx in sss.split(np.zeros(len(labels)), labels):
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = SimpleCNN(num_classes=num_classes).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_loss = running_loss / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}", end="")

    # Save best model
    if val_acc > best_val_acc or epoch == EPOCHS:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        print(" <-", end="")
    print()

model_path = os.path.join(MODELS_DIR, f'georgian_cnn_{best_val_acc}.pth')
torch.save(best_model_state, model_path)
print(f"Saved new best model to {model_path}")

print("Training complete. Best Val Acc: {:.4f}".format(best_val_acc)) 