"""
Medium-Scale Training Script (50 fonts, 50 epochs)
Optimized for Google Colab - good balance of speed and results
Estimated time: 2-4 hours on Colab T4 GPU
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


def create_medium_config():
    """Create config for medium-scale training"""
    return {
        'data': {
            'num_fonts': 50,
            'characters': "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            'image_size': 128,
        },
        'model': {
            'char_vocab_size': 52,
            'max_seq_len': 3,
            'char_embed_dim': 128,
            'font_embed_dim': 128,
            'num_fonts': 50,
            'use_font_attributes': True,
            'font_attr_dim': 8,
            'patch_encoder': {
                'patch_size': 16,
                'in_channels': 1,
                'embed_dim': 384,  # Medium size
            },
            'transformer': {
                'num_layers': 6,   # Medium depth
                'num_heads': 6,
                'dim': 384,
                'mlp_ratio': 4,
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'use_rmsnorm': True,
                'use_swiglu': True,
                'use_rotary_emb': True,
            },
            'decoder': {
                'type': 'spatial_cnn',
                'hidden_dims': [192, 96, 48],
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
            'batch_size': 16,  # Good for Colab
            'num_epochs': 50,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'scheduler': 'cosine',
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 2,  # Effective batch = 32
            'use_amp': True,
            'save_every': 10,
            'eval_every': 5,
            'log_every': 50
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'typo-medium',
            'log_gradients': False,
            'log_weights': False,
            'log_attention_maps': True,
            'log_activations_summary': True,
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medium-scale training')
    parser.add_argument('--data_dir', type=str, default='./data/rendered')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_medium')
    parser.add_argument('--run_name', type=str, default='medium_50fonts_50epochs')
    parser.add_argument('--no_wandb', action='store_true')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MEDIUM-SCALE TRAINING (50 fonts, 50 epochs)")
    logger.info("Optimized for Google Colab T4 GPU")
    logger.info("Estimated time: 2-4 hours")
    logger.info("=" * 80)
    
    # Save config
    config = create_medium_config()
    config_path = Path('config_medium.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"\nSaved config to {config_path}")
    
    # Prepare arguments for main training script
    sys.argv = [
        'train.py',
        '--config', str(config_path),
        '--data_dir', args.data_dir,
        '--checkpoint_dir', args.checkpoint_dir,
        '--run_name', args.run_name,
        '--num_workers', '2'  # Limited for Colab
    ]
    
    if args.no_wandb:
        sys.argv.append('--no_wandb')
    
    # Run training
    train_main()
    
    logger.info("\n" + "=" * 80)
    logger.info("MEDIUM-SCALE TRAINING COMPLETED!")
    logger.info("=" * 80)
