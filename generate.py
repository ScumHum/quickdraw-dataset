#!/usr/bin/env python3
"""
Inference script for generating doodles with the trained model
"""
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
import os

from model import ConditionalVAE


def load_trained_model(model_path, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ConditionalVAE(
        latent_dim=checkpoint['latent_dim'],
        num_categories=checkpoint['num_categories']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['category_names']


def generate_doodles(model, category_names, categories=None, num_samples=4, device='cpu'):
    """
    Generate doodles for specified categories.
    
    Args:
        model: Trained VAE model
        category_names: List of category names
        categories: List of category names or indices to generate. If None, generates for all.
        num_samples: Number of samples to generate per category
        device: Device to run inference on
    
    Returns:
        generated_images: numpy array of generated images
        used_categories: list of category names used for generation
    """
    if categories is None:
        # Generate for all categories
        category_indices = list(range(len(category_names)))
        used_categories = category_names
    else:
        # Parse categories (can be names or indices)
        category_indices = []
        used_categories = []
        
        for cat in categories:
            if isinstance(cat, str):
                if cat in category_names:
                    idx = category_names.index(cat)
                    category_indices.append(idx)
                    used_categories.append(cat)
                else:
                    print(f"Warning: Category '{cat}' not found in model. Available: {category_names}")
            elif isinstance(cat, int):
                if 0 <= cat < len(category_names):
                    category_indices.append(cat)
                    used_categories.append(category_names[cat])
                else:
                    print(f"Warning: Category index {cat} out of range [0, {len(category_names)-1}]")
    
    if not category_indices:
        raise ValueError("No valid categories specified")
    
    all_images = []
    
    for cat_idx in category_indices:
        # Generate samples for this category
        generated = model.generate(cat_idx, num_samples=num_samples, device=device)
        generated = generated.cpu().numpy()
        all_images.append(generated)
    
    # Combine all generated images
    generated_images = np.concatenate(all_images, axis=0)
    
    return generated_images, used_categories


def save_generated_doodles(images, category_names, num_samples, output_dir='generated'):
    """Save generated doodles as individual PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    
    img_idx = 0
    for cat_idx, category in enumerate(category_names):
        for sample_idx in range(num_samples):
            # Get image and convert to 0-255 range
            img = images[img_idx, 0] * 255
            img = img.astype(np.uint8)
            
            # Save as PNG
            pil_img = Image.fromarray(img, mode='L')
            filename = f"{category}_sample_{sample_idx+1}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath)
            
            img_idx += 1
    
    print(f"Saved {len(images)} generated doodles to {output_dir}/")


def visualize_generation_grid(images, category_names, num_samples, save_path='generated_grid.png'):
    """Create a grid visualization of generated doodles."""
    num_categories = len(category_names)
    
    fig, axes = plt.subplots(num_categories, num_samples, 
                            figsize=(num_samples * 2, num_categories * 2))
    
    if num_categories == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    img_idx = 0
    for cat_idx in range(num_categories):
        for sample_idx in range(num_samples):
            ax = axes[cat_idx, sample_idx]
            ax.imshow(images[img_idx, 0], cmap='gray')
            
            if sample_idx == 0:
                ax.set_ylabel(category_names[cat_idx], fontsize=12)
            if cat_idx == 0:
                ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            img_idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grid visualization saved to {save_path}")


def interpolate_between_categories(model, category_names, cat1, cat2, steps=5, device='cpu'):
    """Generate interpolations between two categories in latent space."""
    # Get category indices
    if isinstance(cat1, str):
        cat1_idx = category_names.index(cat1)
    else:
        cat1_idx = cat1
        cat1 = category_names[cat1]
    
    if isinstance(cat2, str):
        cat2_idx = category_names.index(cat2)
    else:
        cat2_idx = cat2
        cat2 = category_names[cat2]
    
    # Generate random latent codes
    z1 = torch.randn(1, model.latent_dim).to(device)
    z2 = torch.randn(1, model.latent_dim).to(device)
    
    interpolations = []
    category_interpolations = []
    
    for i in range(steps):
        # Linear interpolation in latent space
        alpha = i / (steps - 1)
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # For categories, we'll switch at the midpoint
        if alpha < 0.5:
            cat_idx = cat1_idx
            cat_name = cat1
        else:
            cat_idx = cat2_idx
            cat_name = cat2
        
        # Generate image
        with torch.no_grad():
            generated = model.decode(z_interp, torch.tensor([cat_idx]).to(device))
            interpolations.append(generated.detach().cpu().numpy())
        category_interpolations.append(cat_name)
    
    return np.concatenate(interpolations, axis=0), category_interpolations


def main():
    parser = argparse.ArgumentParser(description='Generate doodles with trained model')
    parser.add_argument('--model-path', default='quickdraw_vae.pth', 
                       help='Path to trained model')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Categories to generate (names or indices). If not specified, generates all.')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per category')
    parser.add_argument('--output-dir', default='generated',
                       help='Directory to save generated images')
    parser.add_argument('--interpolate', nargs=2, default=None,
                       help='Generate interpolation between two categories')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available categories and exit')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    try:
        model, category_names = load_trained_model(args.model_path, device)
        print(f"Loaded model with {len(category_names)} categories")
    except FileNotFoundError:
        print(f"Model file {args.model_path} not found. Please train a model first.")
        return
    
    # List categories if requested
    if args.list_categories:
        print("Available categories:")
        for i, cat in enumerate(category_names):
            print(f"  {i}: {cat}")
        return
    
    # Generate interpolation if requested
    if args.interpolate:
        cat1, cat2 = args.interpolate
        print(f"Generating interpolation between '{cat1}' and '{cat2}'...")
        
        try:
            images, cat_names = interpolate_between_categories(
                model, category_names, cat1, cat2, device=device
            )
            
            # Visualize interpolation
            fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
            for i, (img, cat_name) in enumerate(zip(images, cat_names)):
                axes[i].imshow(img[0], cmap='gray')
                axes[i].set_title(cat_name)
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig('interpolation.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("Interpolation saved to interpolation.png")
            
        except ValueError as e:
            print(f"Error generating interpolation: {e}")
        
        return
    
    # Generate doodles
    print(f"Generating doodles...")
    if args.categories:
        print(f"Categories: {args.categories}")
    else:
        print("Generating for all categories")
    
    try:
        images, used_categories = generate_doodles(
            model, category_names, args.categories, args.num_samples, device
        )
        
        print(f"Generated {len(images)} doodles for categories: {used_categories}")
        
        # Save individual images
        save_generated_doodles(images, used_categories, args.num_samples, args.output_dir)
        
        # Create grid visualization
        visualize_generation_grid(images, used_categories, args.num_samples)
        
    except Exception as e:
        print(f"Error generating doodles: {e}")


if __name__ == "__main__":
    main()