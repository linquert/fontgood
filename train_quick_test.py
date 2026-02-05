"""
Quick Test Training Script
Tests the entire pipeline with minimal data (10 fonts, 5 epochs)
Perfect for debugging and verifying code before large-scale training
"""

import torch
import yaml
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train import Trainer
from dataset import TypographicDataset, collate_fn, load_dataset_split
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create minimal config for quick testing"""
    return {
        'data': {
            'num_fonts': 10,
            'characters': "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            'image_size': 128,
        },
        'model': {
            'char_vocab_size': 52,
            'max_seq_len': 3,
            'char_embed_dim': 128,
            'font_embed_dim': 128,
            'num_fonts': 10,  # Will be updated
            'use_font_attributes': True,
            'font_attr_dim': 8,
            'patch_encoder': {
                'patch_size': 16,
                'in_channels': 1,
                'embed_dim': 256,  # Reduced for testing
            },
            'transformer': {
                'num_layers': 4,  # Reduced
                'num_heads': 4,   # Reduced
                'dim': 256,       # Reduced
                'mlp_ratio': 4,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'use_rmsnorm': True,
                'use_swiglu': True,
                'use_rotary_emb': True,
            },
            'decoder': {
                'type': 'spatial_cnn',
                'hidden_dims': [128, 64],  # Reduced
                'output_channels': 1
            },
            'use_vae': False
        },
        'training': {
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.5,
                'font_attr_prediction': 0.1
            },
            'batch_size': 4,  # Small batch
            'num_epochs': 5,  # Quick test
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'use_amp': True,
            'save_every': 2,
            'eval_every': 1,
            'log_every': 10
        },
        'logging': {
            'use_wandb': False,  # Disable for quick test
            'project_name': 'typo-test',
            'log_gradients': False,
            'log_weights': False,
            'log_attention_maps': False,
            'log_activations_summary': False,
        }
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test training')
    parser.add_argument('--data_dir', type=str, default='./data/rendered',
                       help='Path to rendered data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_test',
                       help='Directory to save checkpoints')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("QUICK TEST TRAINING - Small Scale")
    logger.info("=" * 80)
    
    # Create test config
    config = create_test_config()
    
    # Load datasets (use minimal data)
    logger.info("\nLoading datasets...")
    train_fonts = load_dataset_split(args.data_dir, 'train')
    val_fonts = load_dataset_split(args.data_dir, 'val')
    
    if train_fonts is None or val_fonts is None:
        logger.error("Failed to load datasets. Run prepare_data.py first.")
        return
    
    # Use only first 10 fonts for testing
    train_fonts = train_fonts[:10]
    val_fonts = val_fonts[:5]
    
    logger.info(f"Using {len(train_fonts)} train fonts, {len(val_fonts)} val fonts")
    
    # Update config
    config['model']['num_fonts'] = len(train_fonts) + len(val_fonts)
    
    # Create datasets
    train_dataset = TypographicDataset(train_fonts, config['data']['characters'])
    val_dataset = TypographicDataset(val_fonts, config['data']['characters'])
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # No multiprocessing for debugging
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create simple args object
    class SimpleArgs:
        def __init__(self):
            self.checkpoint_dir = args.checkpoint_dir
            self.run_name = 'test_run'
            self.no_wandb = True
            self.wandb_offline = False
            self.num_workers = 0
            self.resume = None
    
    simple_args = SimpleArgs()
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(config, simple_args)
    
    # Train
    logger.info("\nStarting training...")
    logger.info("This is a quick test - should complete in a few minutes")
    
    trainer.train(train_loader, val_loader)
    
    logger.info("\n" + "=" * 80)
    logger.info("QUICK TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("Next: Run full-scale training with train_medium.py or train_full.py")


if __name__ == "__main__":
    main()
