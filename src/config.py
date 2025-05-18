"""
Configuration and constants for the Georgian Character Recognition project.
"""
import os
import torch

# Georgian alphabet
GEORGIAN_ALPHABET = [
    'ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ',
    'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ'
]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Ensure directories exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Image processing
IMAGE_SIZE = 32

# Training parameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 