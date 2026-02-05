"""
Full-Scale Training Script (500 fonts, 150 epochs)
For serious mechanistic interpretability research
Requires good GPU (V100/A100) and ~24 hours
"""

import torch
import yaml
from pathlib import Path
import sys
import logging
import argparse

sys.path.append(str(Path(__file__).parent))

from train import main as train_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_full_config():
    """Create config for full-scale training"""
    return {
        'data': {
            'num_fonts': 500,
            'characters': "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            'image_size': 128,
        },
        'model': {
            'char_vocab_size': 52,
            'max_seq_len': 3,
            'char_embed_dim': 128,
            'font_embed_dim': 128,
            'num_fonts': 500,
            'use_font_attributes': True,
            'font_attr_dim': 8,
            'patch_encoder': {
                'patch_size': 16,
                'in_channels': 1,
                'embed_dim': 512,  # Full size
            },
            'transformer': {
                'num_layers': 8,   # Full depth
                'num_heads': 8,
                'dim': 512,
                'mlp_ratio': 4,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'use_rmsnorm': True,
                'use_swiglu': True,
                'use_rotary_emb': True,
            },
            'decoder': {
                'type': 'spatial_cnn',
                'hidden_dims': [256, 128, 64],
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
            'batch_size': 32,  # Full batch
            'num_epochs': 150,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 2000,
            'scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'use_amp': True,
            'save_every': 10,
            'eval_every': 5,
            'log_every': 100
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'typo-full',
            'log_gradients': False,
            'log_weights': False,
            'log_attention_maps': True,
            'log_activations_summary': True,
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full-scale training')
    parser.add_argument('--data_dir', type=str, default='./data/rendered')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_full')
    parser.add_argument('--run_name', type=str, default='full_500fonts_150epochs')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FULL-SCALE TRAINING (500 fonts, 150 epochs)")
    logger.info("For serious mechanistic interpretability research")
    logger.info("Requires V100/A100 GPU and ~24 hours")
    logger.info("=" * 80)
    
    # Save config
    config = create_full_config()
    config_path = Path('config_full.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"\nSaved config to {config_path}")
    
    # Prepare arguments
    sys.argv = [
        'train.py',
        '--config', str(config_path),
        '--data_dir', args.data_dir,
        '--checkpoint_dir', args.checkpoint_dir,
        '--run_name', args.run_name,
        '--num_workers', '4'
    ]
    
    if args.no_wandb:
        sys.argv.append('--no_wandb')
    
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    # Run training
    train_main()
    
    logger.info("\n" + "=" * 80)
    logger.info("FULL-SCALE TRAINING COMPLETED!")
    logger.info("=" * 80)
