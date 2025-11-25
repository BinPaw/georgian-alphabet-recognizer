# Georgian Character Recognition

A machine learning project for recognizing handwritten Georgian (Mkhedruli) alphabet characters.

## Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten Georgian alphabet characters. It includes functionality for:

- Training a CNN model on handwritten Georgian characters
- Evaluating model performance
- Drawing and saving new character samples
- Live prediction of handwritten characters

## Project Structure

```
georgian-character-recognition/
├── src/                         # Source code
│   ├── models/                  # Model definitions
│   ├── utils/                   # Utility functions
│   ├── training/                # Training scripts
│   ├── config.py                # Configuration and constants
│   └── dataset.py               # Dataset loading and preprocessing
├── scripts/                     # Executable scripts
├── data_original/               # Dataset directory
├── checkpoints/                 # Saved model checkpoints
└── tests/                       # Unit tests
```

## Usage

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Generate augmented data:
```bash
python scripts/generate_augmented_data.py
```

3. Train the model:
```
python scripts/train_model.py
```

4. Run live prediction:
```
python scripts/live_predict.py
```

## Dataset

The dataset in `data_original` directory consists of handwritten Georgian alphabet characters organized in directories by character.

## Model Architecture

The model is a simple CNN with the following architecture:
- 2 convolutional layers with ReLU activation and max pooling
- Fully connected layers with 128 hidden units
- Output layer with 33 classes (one for each Georgian character)

## Data Augmentation
The `generate_augmented_data` script will:
1. Copy all original images from `data_original` to a new `data` directory
2. Generate 5 translated versions of each image
3. Generate 5 rotated versions of each image
4. Save all images in the same character directory structure

### Augmentation Techniques Used:
1. **Translation**: Moves the image randomly in X and Y directions
2. **Rotation**: Applies slight rotations (±15°) to the image

### Benefits of Data Augmentation:
1. Increase the size of the training dataset
2. Improve model generalization by exposing it to variations
3. Reduce overfitting by creating more diverse examples
4. Make the model more robust to different input conditions
5. It is particularly useful for handwritten letters, as it simulates natural variations of writing letters by hand