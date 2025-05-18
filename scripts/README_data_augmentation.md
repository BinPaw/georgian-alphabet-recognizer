# Data Augmentation for Georgian Character Dataset

This directory contains scripts for data augmentation to expand the Georgian character dataset.

## Usage

Before running the script, ensure you have:
1. Renamed the original data directory to `data_original`
2. Installed required dependencies (PIL/Pillow)

To generate augmented data:
```bash
python scripts/generate_augmented_data.py
```

The script will:
1. Copy all original images to the new `data` directory
2. Generate 5 translated versions of each image
3. Generate 5 rotated versions of each image
4. Save all images in the same character directory structure

## Augmentation Techniques Used

The script implements the following data augmentation techniques:

1. **Translation**: Moves the image randomly in X and Y directions
2. **Rotation**: Applies slight rotations (±15°) to the image

## Benefits of Data Augmentation

Data augmentation helps:
1. Increase the size of the training dataset
2. Improve model generalization by exposing it to variations
3. Reduce overfitting by creating more diverse examples
4. Make the model more robust to different input conditions

For handwritten character recognition, translation and rotation augmentations are particularly useful as they simulate the natural variations in handwriting styles, positioning, and image capture conditions.
