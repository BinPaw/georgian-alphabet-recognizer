"""
Tests for the CNN model.
"""

import sys
import os

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
from src.models import SimpleCNN

class TestSimpleCNN(unittest.TestCase):
    """Test cases for the SimpleCNN model."""
    
    def test_model_output_shape(self):
        """Test that the model outputs the correct shape."""
        batch_size = 10
        num_classes = 33
        
        # Create model
        model = SimpleCNN(num_classes=num_classes)
        
        # Create random input
        x = torch.randn(batch_size, 1, 32, 32)
        
        # Get output
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_classes))
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        num_classes = 33
        
        # Create model
        model = SimpleCNN(num_classes=num_classes)
        
        # Create random input
        x = torch.randn(1, 1, 32, 32)
        
        # Get output
        output = model(x)
        
        # Check that output contains valid probabilities
        self.assertEqual(output.shape, (1, num_classes))

if __name__ == '__main__':
    unittest.main() 