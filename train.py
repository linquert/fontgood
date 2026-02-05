"""
Training Script for Pure Typographic Transformer
OPTIMIZED VERSION - All critical bugs fixed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import time
import os
import signal
import sys
from contextlib import nullcontext

# Import project modules - FIXED imports
from typo_transformer import TypographicTransformer
from dataset import TypographicDataset, collate_fn, load_dataset_split
from losses import CombinedLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPrefetcher:
    """Prefetch next batch to GPU while current batch is processing"""
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = None
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_batch = None
    
    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        return self
    
    def preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._move_to_device()
        else:
            self._move_to_device()
    
    def _move_to_device(self):
        for key in self.next_batch:
            if isinstance(self.next_batch[key], torch.Tensor):
                self.next_batch[key] = self.next_batch[key].to(
                    self.device, non_blocking=True
                )
    
    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        self.preload()
        return batch


class EMA:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class Trainer:
    """Complete training pipeline with all optimizations"""
    
    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled TF32 and cuDNN autotuning")
        
        # Setup paths
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = TypographicTransformer(config).to(self.device)
        
        # Enable gradient checkpointing if configured
        if config['model'].get('use_gradient_checkpointing', False):
            logger.info("Gradient checkpointing enabled")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")
        
        # Loss function - FIXED: pass device
        self.criterion = CombinedLoss(config, device=self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Scheduler - FIXED: Don't create until we have train_loader
        self.scheduler = None
        
        # EMA
        if config['training'].get('use_ema', True):
            self.ema = EMA(self.model, decay=0.9999)
            logger.info("Using EMA")
        else:
            self.ema = None
        
        # Mixed precision
        self.use_amp = config['training']['use_amp'] and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training")
        
        # Gradient accumulation
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # WandB
        if config['logging']['use_wandb'] and not args.no_wandb:
            wandb.init(
                project=config['logging']['project_name'],
                name=args.run_name,
                config=config,
                mode='online' if not args.wandb_offline else 'offline'
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals"""
        logger.warning("Interrupt detected! Saving checkpoint...")
        self.save_checkpoint(
            self.current_epoch,
            self.best_val_loss,
            is_best=False,
            emergency=True
        )
        if self.use_wandb:
            wandb.finish()
        sys.exit(0)
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler - FIXED"""
        config = self.config['training']
        
        if config['scheduler'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        elif config['scheduler'] == 'cosine_warmup':
            from torch.optim.lr_scheduler import LambdaLR
            
            warmup_steps = config['warmup_steps']
            total_steps = num_training_steps
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch - OPTIMIZED"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'perceptual': 0.0,
            'font_attr': 0.0
        }
        
        # Use prefetcher for async GPU transfer
        if self.device.type == 'cuda':
            prefetcher = DataPrefetcher(train_loader, self.device)
            pbar = tqdm(prefetcher, desc=f"Epoch {epoch}", total=len(train_loader))
        else:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device (if not using prefetcher)
            if self.device.type != 'cuda':
                images = batch['image'].to(self.device)
                char_indices = batch['char_indices'].to(self.device)
                font_idx = batch['font_idx'].to(self.device)
                font_attrs = batch['font_attrs'].to(self.device)
            else:
                # Already on device from prefetcher
                images = batch['image']
                char_indices = batch['char_indices']
                font_idx = batch['font_idx']
                font_attrs = batch['font_attrs']
            
            # Check if we're accumulating gradients
            is_accumulating = (batch_idx + 1) % self.grad_accum_steps != 0
            
            # Use no_sync context for DDP compatibility and efficiency
            with self.model.no_sync() if is_accumulating and hasattr(self.model, 'no_sync') else nullcontext():
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        output = self.model(
                            image=images,
                            char_indices=char_indices,
                            font_idx=font_idx
                        )
                        
                        losses = self.criterion(
                            pred_image=output['reconstructed'],
                            target_image=images,
                            pred_font_attrs=output.get('font_attrs_pred'),
                            target_font_attrs=font_attrs,
                            epoch=epoch  # For loss warmup
                        )
                        
                        loss = losses['total'] / self.grad_accum_steps
                else:
                    output = self.model(
                        image=images,
                        char_indices=char_indices,
                        font_idx=font_idx
                    )
                    
                    losses = self.criterion(
                        pred_image=output['reconstructed'],
                        target_image=images,
                        pred_font_attrs=output.get('font_attrs_pred'),
                        target_font_attrs=font_attrs,
                        epoch=epoch
                    )
                    
                    loss = losses['total'] / self.grad_accum_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update weights (with gradient accumulation)
            if not is_accumulating:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)  # More efficient
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                # Step scheduler if using step-based scheduler
                if self.scheduler is not None and self.config['training']['scheduler'] == 'cosine_warmup':
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Track losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Update progress bar
            postfix = {
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add GPU memory if available
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                postfix['gpu_gb'] = f'{gpu_mem:.2f}'
            
            pbar.set_postfix(postfix)
            
            # Log to wandb
            if self.use_wandb and self.global_step % self.config['training']['log_every'] == 0:
                log_dict = {
                    'train/loss': losses['total'].item(),
                    'train/reconstruction': losses['reconstruction'].item(),
                    'train/perceptual': losses['perceptual'].item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step
                }
                
                if losses.get('font_attr', 0) > 0:
                    log_dict['train/font_attr'] = losses['font_attr'].item()
                
                if torch.cuda.is_available():
                    log_dict['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
                    log_dict['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                
                wandb.log(log_dict, step=self.global_step)
            
            # Emergency checkpoint every N batches (for Colab)
            if batch_idx % 500 == 0 and batch_idx > 0:
                self.save_checkpoint(
                    epoch, self.best_val_loss, is_best=False, emergency=True
                )
            
            # Clear activations to prevent memory leaks
            if hasattr(self.model, 'blocks'):
                for block in self.model.blocks:
                    if hasattr(block, 'clear_activations'):
                        block.clear_activations()
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validation pass - OPTIMIZED with EMA"""
        # Apply EMA weights if using
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'perceptual': 0.0,
            'font_attr': 0.0
        }
        
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            char_indices = batch['char_indices'].to(self.device)
            font_idx = batch['font_idx'].to(self.device)
            font_attrs = batch['font_attrs'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(
                        image=images,
                        char_indices=char_indices,
                        font_idx=font_idx
                    )
                    
                    losses = self.criterion(
                        pred_image=output['reconstructed'],
                        target_image=images,
                        pred_font_attrs=output.get('font_attrs_pred'),
                        target_font_attrs=font_attrs,
                        epoch=epoch
                    )
            else:
                output = self.model(
                    image=images,
                    char_indices=char_indices,
                    font_idx=font_idx
                )
                
                losses = self.criterion(
                    pred_image=output['reconstructed'],
                    target_image=images,
                    pred_font_attrs=output.get('font_attrs_pred'),
                    target_font_attrs=font_attrs,
                    epoch=epoch
                )
            
            for key in val_losses.keys():
                if key in losses:
                    val_losses[key] += losses[key].item()
            
            num_batches += 1
        
        # Average
        for key in val_losses.keys():
            val_losses[key] /= num_batches
        
        # Restore original weights if using EMA
        if self.ema is not None:
            self.ema.restore()
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {
                'val/loss': val_losses['total'],
                'val/reconstruction': val_losses['reconstruction'],
                'val/perceptual': val_losses['perceptual'],
                'epoch': epoch
            }
            
            if val_losses['font_attr'] > 0:
                log_dict['val/font_attr'] = val_losses['font_attr']
            
            wandb.log(log_dict, step=self.global_step)
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, emergency: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save different types of checkpoints
        if emergency:
            checkpoint_path = self.checkpoint_dir / 'emergency_checkpoint.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Always save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop - FIXED"""
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # FIXED: Create scheduler after we have train_loader
        if self.scheduler is None:
            total_steps = len(train_loader) * self.config['training']['num_epochs']
            self.scheduler = self._create_scheduler(total_steps)
            if self.scheduler is not None:
                logger.info(f"Created scheduler with {total_steps} total steps")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % self.config['training']['eval_every'] == 0:
                val_losses = self.validate(val_loader, epoch)
                
                logger.info(f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f}, "
                          f"Val Loss: {val_losses['total']:.4f}")
                
                # Save checkpoint
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
                if epoch % self.config['training']['save_every'] == 0:
                    self.save_checkpoint(epoch, val_losses['total'], is_best)
            
            # Learning rate scheduling (epoch-based schedulers)
            if self.scheduler is not None and self.config['training']['scheduler'] == 'cosine':
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        logger.info("Training completed!")
        
        # Final save
        self.save_checkpoint(
            self.config['training']['num_epochs'] - 1,
            self.best_val_loss,
            is_best=False
        )
        
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Typographic Transformer')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_dir', type=str, default='./data/rendered')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_fonts = load_dataset_split(args.data_dir, 'train')
    val_fonts = load_dataset_split(args.data_dir, 'val')
    
    if train_fonts is None or val_fonts is None:
        logger.error("Failed to load datasets. Run prepare_data.py first.")
        return
    
    # FIXED: Count all unique fonts across all splits
    all_fonts = set()
    for split in ['train', 'val', 'test']:
        fonts = load_dataset_split(args.data_dir, split)
        if fonts:
            all_fonts.update(f['font_name'] for f in fonts)
    
    config['model']['num_fonts'] = len(all_fonts)
    logger.info(f"Total unique fonts: {len(all_fonts)}")
    
    # Create datasets
    train_dataset = TypographicDataset(
        train_fonts,
        characters=config['data']['characters']
    )
    
    val_dataset = TypographicDataset(
        val_fonts,
        characters=config['data']['characters']
    )
    
    # Create data loaders - OPTIMIZED for Colab
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=False,  # Don't waste memory on Colab
        drop_last=True  # Prevent variable batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=False,
        drop_last=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Trainer(config, args)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()