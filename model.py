#!/usr/bin/env python3
"""
Variational Autoencoder for Quick Draw Sketch Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class QuickDrawDataset(Dataset):
    """Dataset wrapper for Quick Draw data."""
    
    def __init__(self, images, labels=None):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.images[idx], self.labels[idx]
        return self.images[idx]


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for sketch generation.
    Can generate sketches conditioned on category labels.
    """
    
    def __init__(self, latent_dim=128, num_categories=15):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        
        # Embedding for category conditioning
        self.category_embedding = nn.Embedding(num_categories, 32)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 7x7 -> 4x4
            nn.Flatten(),
        )
        
        # Calculate the size after convolutions
        self.encoder_output_size = 128 * 4 * 4 + 32  # +32 for category embedding
        
        # Latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 32, 128 * 4 * 4)  # +32 for category embedding
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 1, 1),   # 16x16 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),    # 16x16 -> 32x32
            nn.Sigmoid()
        )
        
    def encode(self, x, category):
        """Encode input and category to latent distribution parameters."""
        # Encode image
        encoded = self.encoder(x)
        
        # Encode category
        cat_emb = self.category_embedding(category)
        
        # Concatenate
        encoded = torch.cat([encoded, cat_emb], dim=1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, category):
        """Decode latent vector and category to image."""
        # Encode category
        cat_emb = self.category_embedding(category)
        
        # Concatenate latent and category
        z_cat = torch.cat([z, cat_emb], dim=1)
        
        # Decode
        x = self.decoder_input(z_cat)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder(x)
        
        # Crop to 28x28 if needed
        if x.size(-1) != 28:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        
        return x
    
    def forward(self, x, category):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x, category)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, category)
        
        return recon_x, mu, logvar
    
    def generate(self, category, num_samples=1, device='cpu'):
        """Generate new sketches for given category."""
        self.eval()
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create category tensor
            if isinstance(category, (list, tuple, torch.Tensor)) and len(category) > 1:
                # Multiple categories provided
                if len(category) != num_samples:
                    raise ValueError(f"Number of categories ({len(category)}) must match num_samples ({num_samples})")
                category = torch.tensor(category, dtype=torch.long).to(device)
            else:
                # Single category
                if isinstance(category, torch.Tensor):
                    category = category[0] if len(category) == 1 else category
                if not isinstance(category, int):
                    category = int(category)
                category = torch.full((num_samples,), category, dtype=torch.long).to(device)
            
            # Generate
            generated = self.decode(z, category)
            
        return generated


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with reconstruction and KL divergence terms."""
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


def train_vae(model, train_loader, epochs=50, lr=1e-3, beta=1.0, device='cpu'):
    """Train the VAE model."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)
            data = data.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            
            recon_data, mu, logvar = model(data, labels)
            loss = vae_loss(recon_data, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item() / len(data)})
        
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'vae_checkpoint_epoch_{epoch+1}.pth')
    
    return losses


def visualize_results(model, test_loader, category_names, device='cpu', num_samples=8):
    """Visualize reconstruction and generation results."""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    test_data, test_labels = next(data_iter)
    test_data, test_labels = test_data[:num_samples].to(device), test_labels[:num_samples].to(device)
    test_data = test_data.unsqueeze(1)
    
    with torch.no_grad():
        # Reconstruction
        recon_data, _, _ = model(test_data, test_labels)
        
        # Generation (one sample per category)
        num_categories = min(len(category_names), num_samples)
        gen_categories = list(range(num_categories))
        generated = model.generate(gen_categories, num_samples=num_categories, device=device)
    
    # Plot results
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    # Original images
    for i in range(num_samples):
        axes[0, i].imshow(test_data[i, 0].cpu(), cmap='gray')
        axes[0, i].set_title(f'Original\n{category_names[test_labels[i]]}')
        axes[0, i].axis('off')
    
    # Reconstructed images
    for i in range(num_samples):
        axes[1, i].imshow(recon_data[i, 0].cpu(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    # Generated images
    for i in range(min(num_samples, num_categories)):
        axes[2, i].imshow(generated[i, 0].cpu(), cmap='gray')
        axes[2, i].set_title(f'Generated\n{category_names[gen_categories[i]]}')
        axes[2, i].axis('off')
    
    # Hide unused subplots
    for i in range(num_categories, num_samples):
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('vae_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # This will be used by the training script
    pass