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
│   └── config.py                # Configuration and constants
├── scripts/                     # Executable scripts
├── data/                        # Dataset directory
├── checkpoints/                 # Saved model checkpoints
├── notebooks/                   # Jupyter notebooks
└── tests/                       # Unit tests
```

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python scripts/train_model.py
```

3. Run live prediction:
```
python scripts/live_predict.py
```

## Dataset

The dataset consists of handwritten Georgian alphabet characters organized in directories by character.

## Model Architecture

The model is a simple CNN with the following architecture:
- 2 convolutional layers with ReLU activation and max pooling
- Fully connected layers with 128 hidden units
- Output layer with 33 classes (one for each Georgian character) 