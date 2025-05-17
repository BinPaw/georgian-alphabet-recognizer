"""
Data processing utilities for the Georgian Character Recognition project.
"""
from torchvision import transforms
from src.config import IMAGE_SIZE

def get_transform():
    """
    Returns the standard image transformation pipeline for preprocessing.
    
    Returns:
        torchvision.transforms.Compose: A composition of image transforms
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) 