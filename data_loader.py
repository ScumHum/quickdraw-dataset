#!/usr/bin/env python3
"""
Quick Draw Data Downloader
Downloads a subset of Quick Draw dataset for training the doodle generator.
"""
import os
import requests
import numpy as np
from tqdm import tqdm
import urllib.request


def download_data(categories=None, data_dir="data", max_samples_per_category=10000, use_demo=False):
    """
    Download Quick Draw dataset for specified categories, or create demo data.
    
    Args:
        categories: List of category names to download. If None, downloads a default set.
        data_dir: Directory to save the data
        max_samples_per_category: Maximum number of samples to download per category
        use_demo: If True, creates synthetic demo data instead of downloading
    """
    if categories is None:
        # Default set of interesting categories for doodle generation
        categories = [
            "cat", "dog", "face", "house", "tree", "car", "bird", "fish", 
            "flower", "apple", "sun", "moon", "star", "heart", "smile"
        ]
    
    os.makedirs(data_dir, exist_ok=True)
    
    if use_demo:
        print("Creating demo dataset with synthetic sketches...")
        from create_demo_data import create_demo_dataset
        create_demo_dataset(data_dir, categories[:5])  # Use first 5 categories for demo
        return [os.path.join(data_dir, f"{cat}.npy") for cat in categories[:5]]
    
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    
    downloaded_files = []
    failed_downloads = []
    
    for category in categories:
        print(f"Downloading {category}...")
        filename = f"{category}.npy"
        url = f"{base_url}/{filename}"
        local_path = os.path.join(data_dir, filename)
        
        if os.path.exists(local_path):
            print(f"  {filename} already exists, skipping.")
            downloaded_files.append(local_path)
            continue
            
        try:
            # Download the file
            urllib.request.urlretrieve(url, local_path)
            
            # Load and truncate to max_samples_per_category if needed
            data = np.load(local_path)
            if len(data) > max_samples_per_category:
                data = data[:max_samples_per_category]
                np.save(local_path, data)
                
            downloaded_files.append(local_path)
            print(f"  Downloaded {len(data)} samples for {category}")
            
        except Exception as e:
            print(f"  Failed to download {category}: {e}")
            failed_downloads.append(category)
            if os.path.exists(local_path):
                os.remove(local_path)
    
    # If all downloads failed, offer to create demo data
    if not downloaded_files and failed_downloads:
        print("\nAll downloads failed. This might be due to network restrictions.")
        print("Creating demo dataset with synthetic sketches instead...")
        from create_demo_data import create_demo_dataset
        demo_categories = failed_downloads[:5]  # Use first 5 failed categories
        create_demo_dataset(data_dir, demo_categories)
        downloaded_files = [os.path.join(data_dir, f"{cat}.npy") for cat in demo_categories]
    
    return downloaded_files


def load_dataset(data_dir="data", categories=None):
    """
    Load the downloaded dataset into memory.
    
    Args:
        data_dir: Directory containing the .npy files
        categories: List of categories to load. If None, loads all available.
        
    Returns:
        images: numpy array of shape (N, 28, 28) with pixel values 0-255
        labels: numpy array of shape (N,) with category indices
        category_names: list of category names corresponding to label indices
    """
    if categories is None:
        # Get all .npy files in the data directory
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        categories = [f.replace('.npy', '') for f in npy_files]
    
    all_images = []
    all_labels = []
    category_names = []
    
    for i, category in enumerate(categories):
        filepath = os.path.join(data_dir, f"{category}.npy")
        if os.path.exists(filepath):
            data = np.load(filepath)
            all_images.append(data)
            all_labels.append(np.full(len(data), i))
            category_names.append(category)
            print(f"Loaded {len(data)} samples for {category}")
        else:
            print(f"Warning: {filepath} not found, skipping {category}")
    
    if not all_images:
        raise ValueError("No data files found. Please download data first.")
    
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Reshape to (N, 28, 28) and normalize to [0, 1]
    images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    print(f"Total dataset: {len(images)} images across {len(category_names)} categories")
    
    return images, labels, category_names


if __name__ == "__main__":
    # Download sample data (or create demo data if download fails)
    download_data(use_demo=True)
    
    # Load and display dataset info
    images, labels, categories = load_dataset()
    print(f"Dataset shape: {images.shape}")
    print(f"Categories: {categories}")