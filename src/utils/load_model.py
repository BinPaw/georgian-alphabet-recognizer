import os
import torch

from src.config import CHECKPOINTS_DIR, DEVICE, GEORGIAN_ALPHABET
from src.models.cnn import SimpleCNN

def load_model(model_path):
    """
    Loads the model for evaluating from the specified path.

    If None is passed, it defaults to the model in the default directory.
    """
    if model_path is None:
        model_dirs = [d for d in os.listdir(CHECKPOINTS_DIR) if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))]
        if not model_dirs:
            print(f"No model directories found in {CHECKPOINTS_DIR}")
            return
        
        if 'default' not in model_dirs:
            print(f"No 'default' directory found in {CHECKPOINTS_DIR}")
            return
        
        model_files = [f for f in os.listdir(os.path.join(CHECKPOINTS_DIR, 'default')) 
                       if f.endswith('.pth')]
        
        if not model_files:
            print(f"No model files found in {os.path.join(CHECKPOINTS_DIR, 'default')}")
            return
        
        model_path = os.path.join(CHECKPOINTS_DIR, 'default', model_files[0])
    
    print(f"Using model: {model_path}")
    
    # Load model
    model = SimpleCNN(num_classes=len(GEORGIAN_ALPHABET))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model