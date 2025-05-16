import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
from georgian_common import GEORGIAN_ALPHABET, DATA_DIR, IMAGE_SIZE

CANVAS_SIZE = 128
SAVE_SIZE = IMAGE_SIZE

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Draw and Save Georgian Character')
        self.selected_char = tk.StringVar(value=GEORGIAN_ALPHABET[0])
        self.save_count = 1

        # Dropdown for character selection
        self.char_menu = ttk.Combobox(root, values=GEORGIAN_ALPHABET, textvariable=self.selected_char, state='readonly', font=('Arial', 16))
        self.char_menu.pack(pady=5)
        self.char_menu.bind('<<ComboboxSelected>>', self.update_save_count)

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
        tk.Button(btn_frame, text='Save', command=self.save).pack(side='left', padx=5)

        self.update_save_count()

    def paint(self, event):
        x, y = event.x, event.y
        r = 6  # brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def update_save_count(self, event=None):
        char = self.selected_char.get()
        folder = os.path.join(DATA_DIR, char)
        if not os.path.exists(folder):
            os.makedirs(folder)
        existing = [f for f in os.listdir(folder) if f.endswith('.png')]
        self.save_count = len(existing) + 1

    def save(self):
        char = self.selected_char.get()
        folder = os.path.join(DATA_DIR, char)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Resize to SAVE_SIZE x SAVE_SIZE
        img_resized = self.image.resize((SAVE_SIZE, SAVE_SIZE), Image.LANCZOS)
        filename = f'{char}_{self.save_count:03d}.png'
        path = os.path.join(folder, filename)
        img_resized.save(path)
        messagebox.showinfo('Saved', f'Saved to {path}')
        self.save_count += 1
        self.clear()

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop() 