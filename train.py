#!/usr/bin/env python3
"""
Training script for the Quick Draw Doodle Generator
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import os
import matplotlib.pyplot as plt

from data_loader import download_data, load_dataset
from model import ConditionalVAE, QuickDrawDataset, train_vae, visualize_results


def main():
    parser = argparse.ArgumentParser(description='Train Quick Draw Doodle Generator')
    parser.add_argument('--data-dir', default='data', help='Directory for dataset')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for KL divergence')
    parser.add_argument('--download', action='store_true', help='Download data before training')
    parser.add_argument('--categories', nargs='+', default=None, 
                       help='Categories to download/train on')
    parser.add_argument('--use-demo', action='store_true', 
                       help='Use synthetic demo data instead of downloading')
    parser.add_argument('--model-path', default='quickdraw_vae.pth', help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Download data if requested
    if args.download or args.use_demo:
        print("Downloading/creating data...")
        download_data(categories=args.categories, data_dir=args.data_dir, use_demo=args.use_demo)
    
    # Load dataset
    print("Loading dataset...")
    try:
        images, labels, category_names = load_dataset(data_dir=args.data_dir, categories=args.categories)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("Try running with --download flag to download the data first.")
        return
    
    print(f"Loaded {len(images)} images from {len(category_names)} categories")
    print(f"Categories: {category_names}")
    
    # Create dataset and dataloaders
    dataset = QuickDrawDataset(images, labels)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        num_categories=len(category_names)
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    losses = train_vae(
        model, train_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        beta=args.beta, 
        device=device
    )
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'category_names': category_names,
        'latent_dim': args.latent_dim,
        'num_categories': len(category_names)
    }, args.model_path)
    
    print(f"Model saved to {args.model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(model, test_loader, category_names, device)
    
    print("Training completed!")


if __name__ == "__main__":
    main()