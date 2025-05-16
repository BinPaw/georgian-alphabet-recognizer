import os
from georgian_common import GEORGIAN_ALPHABET, DATA_DIR

def create_folders(base_dir=DATA_DIR):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for char in GEORGIAN_ALPHABET:
        folder_path = os.path.join(base_dir, char)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

if __name__ == "__main__":
    create_folders() 