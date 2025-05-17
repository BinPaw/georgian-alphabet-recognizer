"""
Visualization utilities for the Georgian Character Recognition project.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.config import GEORGIAN_ALPHABET

def plot_sample_images(dataset, num_samples=5, figsize=(15, 3)):
    """
    Plot sample images from the dataset.
    
    Args:
        dataset: The dataset to sample from
        num_samples: Number of samples to display
        figsize: Figure size as (width, height)
    """
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        img = img.numpy().squeeze()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {GEORGIAN_ALPHABET[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_prediction(image, probs, top_k=3):
    """
    Plot an image and its top predicted classes.
    
    Args:
        image: The input image tensor
        probs: Probability distribution over classes
        top_k: Number of top predictions to display
    """
    # Convert image for display
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().squeeze()
    
    # Get top k predictions
    top_probs, top_classes = torch.topk(probs, top_k)
    top_probs = top_probs.cpu().detach().numpy()
    top_classes = top_classes.cpu().detach().numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot image
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Plot predictions
    y_pos = np.arange(top_k)
    ax2.barh(y_pos, top_probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([GEORGIAN_ALPHABET[idx] for idx in top_classes])
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    plt.show() 