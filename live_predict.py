import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import torch
from common import GEORGIAN_ALPHABET, SimpleCNN, get_transform, IMAGE_SIZE

CANVAS_SIZE = 128

def find_latest_model():
    latest_model = None
    latest_time = 0
    
    for root, dirs, files in os.walk('models'):
        for file in files:
            if file.endswith('.pth'):
                file_path = os.path.join(root, file)
                file_time = os.path.getctime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_model = file_path
    
    return latest_model

MODEL_PATH = find_latest_model()
print("MODEL_PATH: ", MODEL_PATH)

# Get the transform from common
transform = get_transform()

class LivePredictApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title('Live Georgian Character Predictor')
        self.prediction_var = tk.StringVar(value='?')

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', cursor='cross')
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # PIL image for saving
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text='Clear', command=self.clear).pack(side='left', padx=5)
        tk.Button(btn_frame, text='Predict', command=self.predict).pack(side='left', padx=5)

        # Prediction label
        pred_label = tk.Label(root, textvariable=self.prediction_var, font=('Arial', 32))
        pred_label.pack(pady=10)

        # Load model
        self.model = SimpleCNN(num_classes=len(GEORGIAN_ALPHABET))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def paint(self, event):
        x, y = event.x, event.y
        r = 6  # brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_var.set('?')

    def predict(self):
        # Preprocess image
        img_resized = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_tensor = transform(img_resized).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_char = GEORGIAN_ALPHABET[pred_idx]
            self.prediction_var.set(pred_char)

if __name__ == '__main__':
    root = tk.Tk()
    app = LivePredictApp(root, MODEL_PATH)
    root.mainloop() 