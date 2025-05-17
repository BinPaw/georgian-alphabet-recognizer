"""
Dataset classes for the Georgian Character Recognition project.
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from src.utils.data import get_transform
from src.config import DATA_DIR

class GeorgianHandwrittenDataset(ImageFolder):
    """
    Dataset for Georgian handwritten characters.
    
    This class extends ImageFolder to load images from directories
    organized by character class.
    """
    def __init__(self, root_dir=DATA_DIR, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset
            transform: Optional transform to be applied to the images
        """
        if transform is None:
            transform = get_transform()
        
        super().__init__(root_dir, transform=transform)
        
        # Map class indices to Georgian alphabet characters
        self.idx_to_char = {v: k for k, v in self.class_to_idx.items()}
    
    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Args:
            index: Index of the item
            
        Returns:
            tuple: (image, label) where label is the index of the target class
        """
        img, label = super().__getitem__(index)
        return img, label 