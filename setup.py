#!/usr/bin/env python3
"""
Quick setup script for the Quick Draw Doodle Generator
"""
import os
import subprocess
import sys


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def download_sample_data():
    """Download a small sample of the dataset."""
    print("Downloading sample data...")
    try:
        from data_loader import download_data
        
        # Download a small set of categories for quick testing
        sample_categories = ["cat", "dog", "face", "house", "tree"]
        download_data(categories=sample_categories, max_samples_per_category=5000)
        print("✓ Sample data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download data: {e}")
        return False


def main():
    print("Quick Draw Doodle Generator Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download sample data
    if not download_sample_data():
        print("Warning: Failed to download sample data. You can download it manually later.")
    
    print("\n" + "=" * 40)
    print("Setup completed! You can now:")
    print("1. Train a model: python train.py --download --epochs 20")
    print("2. Generate doodles: python generate.py --categories cat dog face")
    print("3. See all options: python train.py --help or python generate.py --help")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)