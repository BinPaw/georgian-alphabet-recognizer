import os
from PIL import Image
from torch.utils.data import Dataset
from georgian_common import GEORGIAN_ALPHABET, get_transform, DATA_DIR

class GeorgianHandwrittenDataset(Dataset):
    def __init__(self, root_dir=DATA_DIR):
        self.root_dir = root_dir
        self.transform = get_transform()
        self.samples = []
        self.class_to_idx = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

if __name__ == "__main__":
    # Example usage
    dataset = GeorgianHandwrittenDataset(root_dir=DATA_DIR)
    print(f"Number of samples: {len(dataset)}")
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Image shape: {image.shape}, Label: {label}")
    else:
        print("No samples found. Please ensure your data directory is populated with images.")