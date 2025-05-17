#!/usr/bin/env python
"""
Script to set up the project directory structure.
"""
import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define directories to create
DIRECTORIES = [
    'src',
    'src/models',
    'src/utils',
    'src/training',
    'scripts',
    'notebooks',
    'tests',
    'checkpoints',
    'data'
]

def main():
    """Create the project directory structure."""
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Setting up project structure in {root_dir}")
    
    # Create directories
    for directory in DIRECTORIES:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            print(f"Creating directory: {directory}")
            os.makedirs(dir_path)
        else:
            print(f"Directory already exists: {directory}")
    
    print("\nProject structure set up successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train the model: python scripts/train_model.py")
    print("3. Run live prediction: python scripts/live_predict.py")

if __name__ == '__main__':
    main() 