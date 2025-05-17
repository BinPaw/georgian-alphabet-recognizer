"""
CNN model architecture for Georgian character recognition.
"""
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for character recognition.
    
    Architecture:
    - 2 convolutional layers with ReLU activation and max pooling
    - Fully connected layers with 128 hidden units
    - Output layer with num_classes units
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x 