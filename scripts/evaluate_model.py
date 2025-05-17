#!/usr/bin/env python
"""
Script to evaluate a trained Georgian character recognition model.
"""
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SimpleCNN
from src.dataset import GeorgianHandwrittenDataset
from src.training.train import evaluate_model
from src.config import GEORGIAN_ALPHABET, DEVICE, CHECKPOINTS_DIR, BATCH_SIZE
from src.utils.visualization import plot_sample_images

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to evaluate the model."""
    parser = argparse.ArgumentParser(description='Evaluate Georgian character recognition model')
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--show-samples', action='store_true', help='Show sample images')
    parser.add_argument('--confusion-matrix', action='store_true', help='Plot confusion matrix')
    args = parser.parse_args()
    
    # Find latest model if not specified
    if args.model_path is None:
        model_dirs = [d for d in os.listdir(CHECKPOINTS_DIR) if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))]
        if not model_dirs:
            print(f"No model directories found in {CHECKPOINTS_DIR}")
            return
        
        latest_dir = max(model_dirs)
        model_files = [f for f in os.listdir(os.path.join(CHECKPOINTS_DIR, latest_dir)) 
                       if f.endswith('.pth')]
        
        if not model_files:
            print(f"No model files found in {os.path.join(CHECKPOINTS_DIR, latest_dir)}")
            return
        
        args.model_path = os.path.join(CHECKPOINTS_DIR, latest_dir, model_files[0])
    
    print(f"Using model: {args.model_path}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = GeorgianHandwrittenDataset()
    
    # Show sample images if requested
    if args.show_samples:
        plot_sample_images(dataset, num_samples=10, figsize=(20, 4))
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Load model
    print("Loading model...")
    model = SimpleCNN(num_classes=len(dataset.class_to_idx))
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # Evaluate model
    print("Evaluating model...")
    _, accuracy = evaluate_model(model, data_loader)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Compute predictions for all samples
    if args.confusion_matrix:
        print("Computing predictions for confusion matrix...")
        all_preds = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, GEORGIAN_ALPHABET)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=GEORGIAN_ALPHABET,
            zero_division=0
        ))

if __name__ == '__main__':
    main() 