#!/usr/bin/env python3
"""
Demo script that creates synthetic Quick Draw-style data for testing
"""
import numpy as np
import os
from PIL import Image, ImageDraw
import random


def create_synthetic_sketches(category, num_samples=1000, size=28):
    """Create synthetic sketch data that mimics Quick Draw style."""
    sketches = []
    
    for i in range(num_samples):
        # Create a blank image
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        if category == "cat":
            # Draw a simple cat-like shape
            # Face circle
            center_x, center_y = size//2, size//2
            radius = random.randint(8, 12)
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], outline=255, width=1)
            
            # Ears
            ear_size = random.randint(3, 5)
            draw.polygon([(center_x-radius//2, center_y-radius), 
                         (center_x-radius//2-ear_size, center_y-radius-ear_size),
                         (center_x-radius//2+ear_size, center_y-radius-ear_size)], outline=255)
            draw.polygon([(center_x+radius//2, center_y-radius), 
                         (center_x+radius//2-ear_size, center_y-radius-ear_size),
                         (center_x+radius//2+ear_size, center_y-radius-ear_size)], outline=255)
            
            # Eyes
            draw.ellipse([center_x-4, center_y-2, center_x-2, center_y], fill=255)
            draw.ellipse([center_x+2, center_y-2, center_x+4, center_y], fill=255)
            
        elif category == "dog":
            # Draw a simple dog-like shape
            center_x, center_y = size//2, size//2
            radius = random.randint(8, 12)
            
            # Face circle
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], outline=255, width=1)
            
            # Floppy ears
            ear_width = random.randint(4, 6)
            draw.ellipse([center_x-radius-ear_width//2, center_y-radius//2, 
                         center_x-radius+ear_width//2, center_y+radius//2], outline=255)
            draw.ellipse([center_x+radius-ear_width//2, center_y-radius//2, 
                         center_x+radius+ear_width//2, center_y+radius//2], outline=255)
            
            # Eyes
            draw.ellipse([center_x-4, center_y-2, center_x-2, center_y], fill=255)
            draw.ellipse([center_x+2, center_y-2, center_x+4, center_y], fill=255)
            
            # Nose
            draw.ellipse([center_x-1, center_y+2, center_x+1, center_y+4], fill=255)
            
        elif category == "face":
            # Draw a simple face
            center_x, center_y = size//2, size//2
            radius = random.randint(9, 13)
            
            # Face circle
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], outline=255, width=1)
            
            # Eyes
            draw.ellipse([center_x-5, center_y-3, center_x-3, center_y-1], fill=255)
            draw.ellipse([center_x+3, center_y-3, center_x+5, center_y-1], fill=255)
            
            # Mouth
            mouth_y = center_y + 3
            draw.arc([center_x-4, mouth_y-2, center_x+4, mouth_y+2], 0, 180, fill=255)
            
        elif category == "house":
            # Draw a simple house
            base_y = size - 5
            house_width = random.randint(12, 16)
            house_height = random.randint(8, 12)
            house_x = center_x = size//2
            
            # House base
            draw.rectangle([house_x-house_width//2, base_y-house_height,
                           house_x+house_width//2, base_y], outline=255, width=1)
            
            # Roof
            roof_height = random.randint(4, 6)
            draw.polygon([(house_x-house_width//2, base_y-house_height),
                         (house_x, base_y-house_height-roof_height),
                         (house_x+house_width//2, base_y-house_height)], outline=255)
            
            # Door
            door_width = house_width // 4
            draw.rectangle([house_x-door_width//2, base_y-house_height//2,
                           house_x+door_width//2, base_y], outline=255, width=1)
            
        elif category == "tree":
            # Draw a simple tree
            trunk_x = size//2
            trunk_width = random.randint(2, 4)
            trunk_height = random.randint(6, 8)
            trunk_bottom = size - 3
            
            # Trunk
            draw.rectangle([trunk_x-trunk_width//2, trunk_bottom-trunk_height,
                           trunk_x+trunk_width//2, trunk_bottom], fill=255)
            
            # Leaves (circle)
            leaf_radius = random.randint(6, 10)
            draw.ellipse([trunk_x-leaf_radius, trunk_bottom-trunk_height-leaf_radius,
                         trunk_x+leaf_radius, trunk_bottom-trunk_height+leaf_radius], outline=255, width=1)
            
        else:
            # Default: random scribbles
            num_lines = random.randint(3, 8)
            for _ in range(num_lines):
                x1, y1 = random.randint(2, size-2), random.randint(2, size-2)
                x2, y2 = random.randint(2, size-2), random.randint(2, size-2)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=1)
        
        # Convert to numpy array
        sketch_array = np.array(img)
        sketches.append(sketch_array)
    
    return np.array(sketches)


def create_demo_dataset(data_dir="data", categories=None):
    """Create a demo dataset with synthetic sketches."""
    if categories is None:
        categories = ["cat", "dog", "face", "house", "tree"]
    
    os.makedirs(data_dir, exist_ok=True)
    
    for category in categories:
        print(f"Creating synthetic {category} sketches...")
        sketches = create_synthetic_sketches(category, num_samples=1000)
        
        # Save as .npy file
        filepath = os.path.join(data_dir, f"{category}.npy")
        np.save(filepath, sketches)
        print(f"  Saved {len(sketches)} samples to {filepath}")
    
    print(f"Demo dataset created with {len(categories)} categories")


if __name__ == "__main__":
    create_demo_dataset()