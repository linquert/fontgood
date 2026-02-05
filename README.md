# Typographic Transformer - Complete Implementation

A production-ready implementation of a pure transformer for mechanistic interpretability research on typography.

## 🚀 Quick Start (Google Colab)

### Option 1: All-in-One Colab Pipeline

```python
# In a Colab notebook, run:
!git clone <your-repo>
%cd typo_interp
!python colab_pipeline.py
```

This will:
1. Setup environment
2. Download fonts
3. Render dataset
4. Train model
5. Run interpretability experiments
6. Package results for download

### Option 2: Step-by-Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python prepare_data.py --num_fonts 50

# 3. Choose training scale:

# Quick test (10 fonts, 5 epochs, ~10 min)
python train_quick_test.py

# Medium scale (50 fonts, 50 epochs, ~2-4 hours)
python train_medium.py

# Full scale (500 fonts, 150 epochs, ~24 hours)
python train_full.py

# 4. Run interpretability
python run_interp.py \
    --checkpoint checkpoints/best_model.pt \
    --config config_medium.yaml \
    --experiments probes attention patching reconstruction
```

## 📁 File Structure

```
typo_interp/
├── Core Model Files (from uploaded files)
│   ├── config.yaml              # Configuration
│   ├── font_metadata.py         # Font loading
│   ├── renderer.py              # Image rendering
│   ├── image_coding.py          # Patch encoding/decoding
│   ├── transformer.py           # Transformer components
│   ├── typo_transformer.py      # Main model
│   └── losses.py                # Loss functions
│
├── Data & Training (NEW)
│   ├── dataset.py               # PyTorch dataset
│   ├── train.py                 # Main training loop
│   ├── train_quick_test.py      # Quick test (10 fonts)
│   ├── train_medium.py          # Medium scale (50 fonts)
│   └── train_full.py            # Full scale (500 fonts)
│
├── Interpretability Tools (NEW)
│   ├── probes.py                # Linear probes
│   ├── attention_analysis.py    # Attention patterns
│   ├── patching.py              # Activation patching
│   ├── visualization.py         # All visualizations
│   └── run_interp.py            # Main interp script
│
├── Colab Support (NEW)
│   ├── colab_pipeline.py        # Complete Colab workflow
│   └── requirements.txt         # Dependencies
│
└── README.md                    # This file
```

## 🎯 Training Scales Comparison

| Scale | Fonts | Epochs | Time (T4) | Time (V100) | Best For |
|-------|-------|--------|-----------|-------------|----------|
| Quick | 10 | 5 | 10 min | 5 min | Testing code |
| Medium | 50 | 50 | 2-4 hrs | 1-2 hrs | Colab experiments |
| Full | 500 | 150 | 12-18 hrs | 6-12 hrs | Research |

## 💾 Memory Requirements

- **Quick**: 2GB GPU RAM, 1GB system RAM
- **Medium**: 4GB GPU RAM, 4GB system RAM  
- **Full**: 8GB GPU RAM, 16GB system RAM

Colab free tier (T4 GPU with 15GB RAM) can comfortably run Medium scale.

## 📊 Expected Results

### Probe Accuracies (Medium scale, layer 6)
- Character identity: 95-99%
- Serif vs sans: 90-95%
- Weight category: 85-90%
- Uppercase: 95-98%

### Reconstruction Quality
- MSE: < 0.01 (after 50 epochs)
- Visual quality: Excellent for single chars, good for sequences

### Attention Patterns
- Early layers (0-2): Local features
- Middle layers (3-5): Feature composition
- Late layers (6-7): Style integration

## 🔬 Interpretability Experiments

### 1. Linear Probes
```python
python run_interp.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --experiments probes
```

Trains classifiers on layer activations to detect:
- Character identity (52-way)
- Font attributes (serif, weight, etc.)
- Sequence properties

**Output**: `probe_accuracies.png` showing accuracy by layer

### 2. Attention Analysis
```python
python run_interp.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --experiments attention
```

Analyzes:
- Attention entropy (focus measure)
- Token importance across layers
- Head specialization patterns

**Outputs**:
- `attention_entropy.png`
- `token_importance.png`
- `head_specialization.png`

### 3. Activation Patching
```python
python run_interp.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --experiments patching
```

Tests causal effects by swapping activations between inputs.

**Output**: Layer-wise effect magnitudes

### 4. Reconstruction & Interpolation
```python
python run_interp.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --experiments reconstruction interpolation
```

Visualizes:
- Original vs reconstructed images
- Font interpolation (smooth transitions)

## 🛠️ Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
model:
  transformer:
    num_layers: 6
    num_heads: 8
    dim: 384

training:
  batch_size: 24
  num_epochs: 100
  learning_rate: 5e-4
```

Then train:
```bash
python train.py --config my_config.yaml
```

### Resume Training

```bash
python train.py --resume checkpoints/latest.pt
```

### WandB Integration

```bash
# Enable WandB logging
python train_medium.py --run_name my_experiment

# Disable WandB
python train_medium.py --no_wandb
```

### Data Augmentation

Edit `dataset.py` to enable augmentation:

```python
from dataset import DataAugmentation

aug = DataAugmentation(
    rotation_range=5.0,
    noise_std=0.02
)

dataset = TypographicDataset(fonts, transform=aug)
```

## 📈 Monitoring Training

### With WandB
Visit wandb.ai to see real-time:
- Loss curves
- Attention visualizations
- Sample reconstructions

### Without WandB
Check console output for:
- Per-epoch losses
- Validation metrics
- Training speed

Checkpoints saved to `./checkpoints/`:
- `best_model.pt` - Best validation loss
- `latest.pt` - Most recent
- `checkpoint_epoch_N.pt` - Periodic saves

## 🐛 Troubleshooting

### Out of Memory
1. Reduce batch size in config
2. Enable gradient accumulation
3. Use smaller model (reduce `dim` or `num_layers`)

### Slow Data Loading
1. Reduce `num_workers` (try 0 for debugging)
2. Use cached rendered data
3. Reduce number of sequences

### Poor Reconstruction
1. Train longer
2. Increase perceptual loss weight
3. Check for NaN gradients
4. Reduce learning rate

## 🔧 Development

### Adding New Probes

Edit `probes.py`:

```python
# In extract_activations method:
all_labels['my_probe'] = compute_my_labels(batch)

# In train_all_probes method:
probe_configs.append(('my_probe', labels['my_probe'], num_classes))
```

### Custom Visualizations

Edit `visualization.py`:

```python
class InterpVisualizer:
    def plot_my_analysis(self, data, save_name='my_plot.png'):
        # Your plotting code
        plt.savefig(self.save_dir / save_name)
```
