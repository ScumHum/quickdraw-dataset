# Quick Draw Neural Network Doodle Generator

This project implements a **Conditional Variational Autoencoder (CVAE)** that can generate new doodles of whatever you want using the Quick Draw dataset!

## Features

🎨 **Generate Custom Doodles**: Create new sketches for any of the 345 Quick Draw categories  
🔄 **Category Conditioning**: Generate specific types of doodles (cats, dogs, houses, etc.)  
🌈 **Latent Interpolation**: Create smooth transitions between different categories  
⚡ **Easy Training**: Simple command-line interface for training on custom data  
📊 **Visualization**: Built-in tools for visualizing training progress and results  

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo with synthetic data
python demo.py
```

### 2. Train Your Own Model
```bash
# Create demo data and train
python train.py --use-demo --epochs 20 --batch-size 64

# Or download real Quick Draw data (if available)
python train.py --download --categories cat dog face house tree --epochs 50
```

### 3. Generate Doodles
```bash
# Generate doodles for specific categories
python generate.py --categories cat dog face --num-samples 4

# Generate for all trained categories
python generate.py --num-samples 2

# Create interpolation between categories
python generate.py --interpolate cat dog

# List available categories
python generate.py --list-categories
```

## Model Architecture

The Conditional VAE consists of:

- **Encoder**: Convolutional layers that encode 28×28 sketches into a latent space
- **Category Embedding**: Learned representations for each drawing category  
- **Latent Space**: Compressed representation that captures drawing features
- **Decoder**: Transposed convolutions that generate sketches from latent codes
- **Conditioning**: Category information is injected at both encoding and decoding stages

```
Input (28x28) → Encoder → Latent Space (128D) → Decoder → Output (28x28)
     ↑                          ↑                    ↑
Category Embedding ──────────────┴────────────────────┘
```

## Training Details

- **Loss Function**: Beta-VAE with reconstruction loss (BCE) + KL divergence
- **Optimizer**: Adam with learning rate 1e-3
- **Batch Size**: 32-128 (configurable)
- **Epochs**: 20-50 (configurable)
- **Data Format**: 28×28 grayscale images, normalized to [0,1]

## File Structure

```
quickdraw-dataset/
├── model.py              # Conditional VAE implementation
├── train.py              # Training script
├── generate.py           # Generation and inference script  
├── data_loader.py        # Data download and loading utilities
├── create_demo_data.py   # Synthetic data creation for demos
├── demo.py               # Complete demonstration script
├── setup.py              # Setup and installation script
└── requirements.txt      # Python dependencies
```

## Usage Examples

### Training Examples
```bash
# Quick training with demo data
python train.py --use-demo --epochs 10

# Train on specific categories
python train.py --download --categories apple banana cake pizza --epochs 30

# Custom training parameters
python train.py --use-demo --epochs 20 --batch-size 64 --lr 0.001 --latent-dim 128
```

### Generation Examples
```bash
# Generate 6 samples each for cat and dog
python generate.py --categories cat dog --num-samples 6

# Generate using category indices
python generate.py --categories 0 2 4 --num-samples 3

# Create smooth interpolation
python generate.py --interpolate house tree

# Generate all categories with 2 samples each
python generate.py --num-samples 2
```

## Output Files

- **Individual Images**: `generated/category_sample_N.png`
- **Grid Visualization**: `generated_grid.png` 
- **Interpolation**: `interpolation.png`
- **Training Progress**: `vae_results.png`
- **Loss Curves**: `training_loss.png`
- **Model Checkpoint**: `quickdraw_vae.pth`

## Advanced Features

### Custom Category Training
Train on your own subset of the 345 available categories:

```bash
python train.py --download --categories \
  "cat" "dog" "bird" "fish" "butterfly" "spider" \
  --epochs 40 --batch-size 64
```

### Latent Space Exploration
The model learns a 128-dimensional latent space where similar concepts are close together. You can:

- Generate variations by sampling around a point
- Interpolate between categories
- Explore the latent space structure

### Model Customization
Adjust the model architecture by modifying `model.py`:

- Change `latent_dim` for different compression levels
- Adjust convolutional layers for different image sizes
- Modify the category embedding dimension
- Experiment with different loss functions (β-VAE parameter)

## Troubleshooting

### Data Download Issues
If downloading from Google Cloud Storage fails:
```bash
# Use demo data instead
python train.py --use-demo --epochs 20
```

### Memory Issues
Reduce batch size and/or latent dimension:
```bash
python train.py --batch-size 16 --latent-dim 64
```

### Training Not Converging
Try adjusting the β parameter for better reconstruction/diversity trade-off:
```bash
python train.py --beta 0.5  # Less regularization, better reconstruction
python train.py --beta 2.0  # More regularization, more diversity
```

## Technical Notes

- **Input Format**: 28×28 grayscale images normalized to [0,1]
- **Categories**: Integer labels 0 to N-1 for N categories
- **Latent Space**: 128-dimensional Gaussian distribution
- **Output**: Sigmoid activation for [0,1] pixel values
- **Device Support**: Automatic CPU/GPU detection

## Future Improvements

- Support for variable-length stroke sequences (like original Sketch-RNN)
- GAN-based generation for higher quality output
- Style transfer between categories
- Interactive web interface for real-time generation
- Support for custom drawing categories

---

*Based on the Quick Draw dataset by Google Creative Lab*