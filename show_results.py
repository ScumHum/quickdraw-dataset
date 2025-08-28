#!/usr/bin/env python3
"""
Display generated results
"""
import matplotlib.pyplot as plt
from PIL import Image
import os

def show_results():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show the grid visualization
    if os.path.exists('generated_grid.png'):
        img = Image.open('generated_grid.png')
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Generated Doodles Grid', fontsize=14)
        axes[0, 0].axis('off')
    
    # Show interpolation
    if os.path.exists('interpolation.png'):
        img = Image.open('interpolation.png')
        axes[0, 1].imshow(img)
        axes[0, 1].set_title('Category Interpolation', fontsize=14)
        axes[0, 1].axis('off')
    
    # Show training results
    if os.path.exists('vae_results.png'):
        img = Image.open('vae_results.png')
        axes[1, 0].imshow(img)
        axes[1, 0].set_title('Training Results (Original/Reconstructed/Generated)', fontsize=14)
        axes[1, 0].axis('off')
    
    # Show training loss
    if os.path.exists('training_loss.png'):
        img = Image.open('training_loss.png')
        axes[1, 1].imshow(img)
        axes[1, 1].set_title('Training Loss Curve', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Demo results saved to demo_results.png")

if __name__ == "__main__":
    os.chdir('/home/runner/work/quickdraw-dataset/quickdraw-dataset')
    show_results()