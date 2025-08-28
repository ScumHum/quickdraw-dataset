#!/usr/bin/env python3
"""
Demo script showcasing the Quick Draw Doodle Generator
"""
import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print("Error:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    print("Quick Draw Neural Network Doodle Generator Demo")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir('/home/runner/work/quickdraw-dataset/quickdraw-dataset')
    
    # Step 1: Create demo data if not exists
    if not os.path.exists('data'):
        print("Creating demo dataset...")
        if not run_command("python create_demo_data.py", "Creating synthetic sketch dataset"):
            return False
    
    # Step 2: Quick training (if model doesn't exist)
    if not os.path.exists('quickdraw_vae.pth'):
        print("\nTraining VAE model (this may take a few minutes)...")
        if not run_command("python train.py --epochs 3 --batch-size 32", "Training Conditional VAE"):
            return False
    
    # Step 3: List available categories
    run_command("python generate.py --list-categories", "Listing available categories")
    
    # Step 4: Generate doodles for specific categories
    run_command("python generate.py --categories cat dog face --num-samples 2", 
                "Generating doodles for cat, dog, and face")
    
    # Step 5: Generate interpolation
    run_command("python generate.py --interpolate cat dog", 
                "Creating interpolation between cat and dog")
    
    # Step 6: Generate all categories
    run_command("python generate.py --num-samples 1", 
                "Generating one sample for each category")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("Generated files:")
    print("- Individual doodles: generated/*.png")
    print("- Grid visualization: generated_grid.png") 
    print("- Interpolation: interpolation.png")
    print("- Training visualization: vae_results.png")
    print("- Training loss: training_loss.png")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)