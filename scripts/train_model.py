#!/usr/bin/env python
"""
Script to train a Georgian character recognition model.
"""
import argparse
import os
import sys
import torch

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SimpleCNN
from src.dataset import GeorgianHandwrittenDataset
from src.training.train import prepare_data_loaders, train_model
from src.config import EPOCHS, LEARNING_RATE, BATCH_SIZE, CHECKPOINTS_DIR

def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train Georgian character recognition model')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINTS_DIR, help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Create dataset
    print("Loading dataset...")
    dataset = GeorgianHandwrittenDataset()
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        dataset, batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    num_classes = len(dataset.class_to_idx)
    model = SimpleCNN(num_classes=num_classes)
    
    # Train model
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"Final model saved to: {results['final_model_path']}")

if __name__ == '__main__':
    main() 