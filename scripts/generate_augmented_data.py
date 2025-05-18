import os
import shutil
import random
from PIL import Image

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def translate_image(img):
    """Apply random translation to image."""
    width, height = img.size
    tx = random.randint(-width//10, width//10)
    ty = random.randint(-height//10, height//10)
    return img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=(255, 255, 255))

def rotate_image(img):
    """Apply slight random rotation to image."""
    angle = random.uniform(-15, 15)
    
    # Rotate the image
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    
    return rotated

def main():
    src_dir = "data_original"
    dst_dir = "data"
    
    # Make sure destination directory exists
    ensure_dir(dst_dir)
    
    # List all character directories in the source
    for char_dir in os.listdir(src_dir):
        src_char_path = os.path.join(src_dir, char_dir)
        dst_char_path = os.path.join(dst_dir, char_dir)
        
        if not os.path.isdir(src_char_path):
            continue
        
        # Create character directory in destination
        ensure_dir(dst_char_path)
        
        # Get all images in the character directory
        images = [f for f in os.listdir(src_char_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(images)} images for character {char_dir}")
        
        for img_file in images:
            src_img_path = os.path.join(src_char_path, img_file)
            
            # Copy original image
            dst_img_path = os.path.join(dst_char_path, img_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Read the image with PIL
            try:
                img = Image.open(src_img_path)
                # Convert to RGB mode if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                print(f"Warning: Could not read {src_img_path}: {e}")
                continue
            
            # Generate 5 translated versions
            for i in range(5):
                # Apply translation
                translated_img = translate_image(img)
                
                # Save translated image
                filename, ext = os.path.splitext(img_file)
                aug_filename = f"{filename}_trans{i}{ext}"
                aug_path = os.path.join(dst_char_path, aug_filename)
                translated_img.save(aug_path)
            
            # Generate 5 rotated versions
            for i in range(5):
                # Apply rotation
                rotated_img = rotate_image(img)
                
                # Save rotated image
                filename, ext = os.path.splitext(img_file)
                aug_filename = f"{filename}_rot{i}{ext}"
                aug_path = os.path.join(dst_char_path, aug_filename)
                rotated_img.save(aug_path)

if __name__ == "__main__":
    main() 