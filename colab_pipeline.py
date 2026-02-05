"""
Google Colab Setup and Training Script
Complete pipeline for training on Colab with optimizations
"""

# ============================================================================
# PART 1: Environment Setup
# ============================================================================

print("=" * 80)
print("TYPOGRAPHIC TRANSFORMER - GOOGLE COLAB SETUP")
print("=" * 80)

# Check GPU
import torch
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Install dependencies
print("\n" + "=" * 80)
print("Installing dependencies...")
print("=" * 80)

!pip install -q torch torchvision
!pip install -q lpips einops wandb pyyaml tqdm matplotlib seaborn scikit-learn scipy

print("✓ Dependencies installed")

# ============================================================================
# PART 2: Data Preparation (Optimized for Colab)
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATASET")
print("=" * 80)

# Create directory structure
!mkdir -p data/cache data/rendered checkpoints

# Clone fonts (if not already done)
import os
from pathlib import Path

fonts_path = Path('data/cache/fonts')
if not fonts_path.exists():
    print("\nCloning Google Fonts repository...")
    !cd data/cache && git clone --depth 1 https://github.com/google/fonts.git
    print("✓ Fonts cloned")
else:
    print("✓ Fonts already available")

# ============================================================================
# PART 3: Prepare Data (Fast Version)
# ============================================================================

print("\n" + "=" * 80)
print("RENDERING FONTS")
print("=" * 80)

# Configuration for Colab (medium scale)
COLAB_CONFIG = {
    'num_fonts': 50,  # Good balance for Colab
    'use_cache': True,  # Reuse if available
    'parallel_render': False,  # Disable for stability
}

# Import modules
import sys
sys.path.append('.')

from font_metadata import FontMetadataLoader
from renderer import SequenceRenderer

# Load fonts
print("\n[1/3] Loading fonts...")
loader = FontMetadataLoader('./data/cache')
fonts = loader.load_all_fonts(min_fonts=20, max_fonts=COLAB_CONFIG['num_fonts'])

if len(fonts) < 20:
    print("ERROR: Not enough fonts loaded")
    sys.exit(1)

print(f"✓ Loaded {len(fonts)} fonts")

# Create splits
print("\n[2/3] Creating splits...")
splits = loader.create_splits(
    fonts,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    continual_ratio=0.0  # No continual learning for Colab
)

# Check if data already rendered
rendered_dir = Path('./data/rendered')
train_cache = rendered_dir / 'train_dataset.pkl'

if COLAB_CONFIG['use_cache'] and train_cache.exists():
    print("✓ Using cached rendered data")
else:
    print("\n[3/3] Rendering images...")
    
    renderer = SequenceRenderer(
        image_size=128,
        font_size=48,
        output_dir='./data/rendered'
    )
    
    # Generate sequences (fewer for speed)
    sequences = renderer.generate_sequences(
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        max_length=2,  # Only 1-2 char sequences for speed
        common_bigrams=["Th", "th", "he", "in", "er", "an", "re", "on"],
        common_trigrams=[]  # Skip trigrams for speed
    )
    
    print(f"Rendering {len(sequences)} sequences per font...")
    
    # Render each split
    for split_name, split_fonts in splits.items():
        if split_name == 'continual':
            continue
        
        print(f"\nRendering {split_name}...")
        rendered = renderer.render_dataset(split_fonts, sequences, split_name)
        total_samples = sum(len(f['sequences']) for f in rendered)
        print(f"✓ {split_name}: {len(rendered)} fonts, {total_samples} samples")
    
    print("\n✓ All data rendered and cached")

# ============================================================================
# PART 4: Training
# ============================================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

# Choose training scale
print("\nSelect training scale:")
print("1. Quick test (10 fonts, 5 epochs, ~10 minutes)")
print("2. Medium (50 fonts, 30 epochs, ~2 hours)")
print("3. Full (50 fonts, 100 epochs, ~6 hours)")

SCALE = 2  # Default to medium

if SCALE == 1:
    !python train_quick_test.py --data_dir ./data/rendered --checkpoint_dir ./checkpoints
elif SCALE == 2:
    !python train_medium.py --data_dir ./data/rendered --checkpoint_dir ./checkpoints --run_name colab_medium
else:
    !python train_full.py --data_dir ./data/rendered --checkpoint_dir ./checkpoints --run_name colab_full

# ============================================================================
# PART 5: Interpretability Analysis
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING INTERPRETABILITY ANALYSIS")
print("=" * 80)

# Find best checkpoint
import glob
checkpoints = glob.glob('./checkpoints/best_model.pt')

if checkpoints:
    best_checkpoint = checkpoints[0]
    print(f"\nFound checkpoint: {best_checkpoint}")
    
    # Run interpretability
    !python run_interp.py \
        --checkpoint {best_checkpoint} \
        --config config_medium.yaml \
        --data_dir ./data/rendered \
        --save_dir ./interp_results \
        --experiments probes attention reconstruction interpolation
    
    print("\n✓ Interpretability analysis complete!")
    print("\nView results in ./interp_results/")
else:
    print("No checkpoint found - training may have failed")

# ============================================================================
# PART 6: Download Results
# ============================================================================

print("\n" + "=" * 80)
print("PACKAGING RESULTS")
print("=" * 80)

# Zip results for download
!zip -r results.zip checkpoints/ interp_results/ -q

print("✓ Results packaged to results.zip")
print("\nYou can download:")
print("  - checkpoints/ (trained models)")
print("  - interp_results/ (analysis and visualizations)")
print("  - results.zip (everything)")

from google.colab import files
print("\nDownloading results.zip...")
files.download('results.zip')

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
