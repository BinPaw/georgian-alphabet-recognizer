"""
Training functions for the Georgian Character Recognition project.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import time
from datetime import datetime
from src.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT, CHECKPOINTS_DIR

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, 
                checkpoint_dir=CHECKPOINTS_DIR):
    """
    Train a model on the given data loaders.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train for
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        dict: Training history and best model path
    """
    # Create a timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    run_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = None
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training phase
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
        
        train_loss = running_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # best_model_path = os.path.join(run_dir, f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth')
            # torch.save(model.state_dict(), best_model_path)
            # print(f"Saved new best model to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(run_dir, f'model_final_acc_{val_acc:.4f}.pth')
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    return {
        'history': history,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path
    }

def evaluate_model(model, data_loader, criterion=None):
    """
    Evaluate a model on the given data loader.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function (if None, only accuracy is computed)
        
    Returns:
        tuple: (loss, accuracy) if criterion is provided, else (None, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    if criterion is not None:
        loss = running_loss / total
        return loss, accuracy
    else:
        return None, accuracy

def prepare_data_loaders(dataset, batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """
    Prepare train and validation data loaders from a dataset.
    
    Args:
        dataset: The dataset to split
        batch_size: Batch size for the data loaders
        val_split: Fraction of data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get labels for stratified split
    labels = [label for _, label in dataset.samples]
    labels = np.array(labels)
    
    # Create stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    for train_idx, val_idx in sss.split(np.zeros(len(labels)), labels):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader 