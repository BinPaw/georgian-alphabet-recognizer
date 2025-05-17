#!/usr/bin/env python
"""
Script for drawing and saving Georgian character samples.
This script opens a drawing canvas where users can draw characters
and save them to the dataset directory.
"""
import argparse
import os
import sys
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GEORGIAN_ALPHABET, DATA_DIR

class DrawingApp:
    """
    Drawing application for creating character samples.
    """
    def __init__(self, save_dir, width=400, height=400):
        self.root = tk.Tk()
        self.root.title("Draw Georgian Characters")
        
        # Save directory
        self.save_dir = save_dir
        
        # Drawing area
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Create a PIL image to draw on
        self.image = Image.new("L", (width, height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Character selection
        self.char_frame = tk.Frame(self.control_frame)
        self.char_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(self.char_frame, text="Character:").pack(side=tk.LEFT)
        
        self.char_var = tk.StringVar(self.root)
        self.char_var.set(GEORGIAN_ALPHABET[0])
        
        self.char_menu = tk.OptionMenu(self.char_frame, self.char_var, *GEORGIAN_ALPHABET)
        self.char_menu.config(width=5)
        self.char_menu.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        self.clear_button = tk.Button(self.control_frame, text="Clear", command=self.clear)
        self.clear_button.pack(fill=tk.X, padx=10, pady=5)
        
        self.save_button = tk.Button(self.control_frame, text="Save", command=self.save)
        self.save_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Status
        self.status_var = tk.StringVar(self.root)
        self.status_var.set("Ready")
        
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.BOTTOM, padx=10, pady=10)
        
        # Last mouse position
        self.last_x, self.last_y = None, None
    
    def paint(self, event):
        """Handle mouse movement for drawing."""
        x, y = event.x, event.y
        
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(
                (self.last_x, self.last_y, x, y),
                fill="white",
                width=15,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255,
                width=15,
                joint="curve"
            )
        
        self.last_x, self.last_y = x, y
    
    def clear(self):
        """Clear the canvas and image."""
        self.canvas.delete("all")
        self.image = Image.new("L", self.image.size, color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.status_var.set("Ready")
    
    def save(self):
        """Save the current drawing."""
        # Get selected character
        char = self.char_var.get()
        
        # Create directory if it doesn't exist
        char_dir = os.path.join(self.save_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # Count existing files to generate a new filename
        existing_files = len([f for f in os.listdir(char_dir) if f.endswith('.png')])
        filename = f"{existing_files + 1:04d}.png"
        filepath = os.path.join(char_dir, filename)
        
        # Resize to a smaller image (32x32)
        resized_img = self.image.resize((32, 32), Image.LANCZOS)
        
        # Save the image
        resized_img.save(filepath)
        self.status_var.set(f"Saved to {filepath}")
        
        # Clear for next drawing
        self.clear()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()

def main():
    """Main function to run the drawing app."""
    parser = argparse.ArgumentParser(description='Draw and save Georgian character samples')
    parser.add_argument('--save-dir', type=str, default=DATA_DIR, help='Directory to save character samples')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run app
    app = DrawingApp(args.save_dir)
    app.run()

if __name__ == '__main__':
    main() 