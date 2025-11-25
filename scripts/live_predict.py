#!/usr/bin/env python
"""
Script for live prediction of Georgian characters.
This script opens a drawing canvas where users can draw characters
and get real-time predictions from the trained model.
"""
import argparse
import os
import sys
import torch
import tkinter as tk
from PIL import Image, ImageDraw

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GEORGIAN_ALPHABET, DEVICE
from src.utils.data import get_transform
from src.utils.load_model import load_model

class DrawingApp:
    """
    Drawing application for character recognition.
    """
    def __init__(self, model, transform, width=128, height=128):
        self.root = tk.Tk()
        self.root.title("Georgian Character Recognition")
        
        # Set up model
        self.model = model
        self.transform = transform
        
        # Drawing area
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        
        # Create a PIL image to draw on
        self.image = Image.new("L", (width, height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Prediction display
        self.prediction_frame = tk.Frame(self.root)
        self.prediction_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top predictions
        self.prediction_labels = []
        self.prediction_values = []
        for i in range(5):
            label_frame = tk.Frame(self.prediction_frame)
            label_frame.pack(fill=tk.X, padx=10, pady=5)
            
            char_label = tk.Label(label_frame, text="", font=("Arial", 24))
            char_label.pack(side=tk.LEFT, padx=10)
            
            prob_label = tk.Label(label_frame, text="", width=10)
            prob_label.pack(side=tk.RIGHT)
            
            self.prediction_labels.append(char_label)
            self.prediction_values.append(prob_label)
        
        # Control buttons
        self.control_frame = tk.Frame(self.prediction_frame)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        self.clear_button = tk.Button(self.control_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = tk.Button(self.control_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.RIGHT, padx=10)
        
        # Last mouse position
        self.last_x, self.last_y = None, None
    
    def mouse_down(self, event):
        """Handle mouse button press."""
        self.last_x, self.last_y = event.x, event.y
    
    def mouse_up(self, event):
        """Handle mouse button release."""
        self.last_x, self.last_y = None, None
    
    def paint(self, event):
        """Handle mouse movement for drawing."""
        x, y = event.x, event.y
        
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                (self.last_x, self.last_y, x, y),
                fill="black",
                width=12,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=0,
                width=12,
                joint="curve"
            )
        
        self.last_x, self.last_y = x, y
    
    def clear(self):
        """Clear the canvas and image."""
        self.canvas.delete("all")
        self.image = Image.new("L", self.image.size, color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        
        # Clear predictions
        for label, value in zip(self.prediction_labels, self.prediction_values):
            label.config(text="")
            value.config(text="")
    
    def predict(self):
        """Make a prediction on the current drawing."""
        # Convert PIL image to tensor
        img = self.transform(self.image).unsqueeze(0).to(DEVICE)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Update prediction display
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            char = GEORGIAN_ALPHABET[idx]
            self.prediction_labels[i].config(text=char)
            self.prediction_values[i].config(text=f"{prob.item():.4f}")
    
    def run(self):
        """Run the application."""
        self.root.mainloop()

def main():
    """Main function to run the live prediction app."""
    parser = argparse.ArgumentParser(description='Live prediction of Georgian characters')
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    args = parser.parse_args()
    
    # Load the model from the path
    model = load_model(args.model_path)
    
    # Get transform
    transform = get_transform()
    
    # Run app
    app = DrawingApp(model, transform)
    app.run()

if __name__ == '__main__':
    main() 