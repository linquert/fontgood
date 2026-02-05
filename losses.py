"""
Loss Functions for Pure Typographic Transformer
FIXED VERSION - Proper device handling and optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import lpips


class CombinedLoss(nn.Module):
    """
    Combined loss with proper device handling and optimizations
    
    Fixes:
    - LPIPS device mismatch
    - Optional LPIPS computation frequency
    - Loss warmup support
    """
    
    def __init__(self, config: Dict, device='cuda'):
        super().__init__()
        
        self.device = device
        loss_weights = config['training']['loss_weights']
        
        # Reconstruction loss (MSE)
        self.recon_weight = loss_weights['reconstruction']
        
        # Perceptual loss (LPIPS) - FIXED: proper device handling
        self.perceptual_weight = loss_weights['perceptual']
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()
        
        # Freeze LPIPS parameters
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        # OPTIMIZATION: Compute LPIPS less frequently to speed up training
        self.lpips_every_n_batches = config['training'].get('lpips_every_n_batches', 1)
        self.batch_counter = 0
        
        # Loss warmup epochs (optional)
        self.perceptual_warmup_epochs = config['training'].get('perceptual_warmup_epochs', 0)
        
        # Font attribute prediction (auxiliary)
        self.use_font_attrs = config['model']['use_font_attributes']
        if self.use_font_attrs:
            self.font_attr_weight = loss_weights['font_attr_prediction']
        
    def get_perceptual_weight(self, epoch: int = 0) -> float:
        """
        Get perceptual loss weight with optional warmup
        
        Args:
            epoch: Current training epoch
        
        Returns:
            Adjusted weight
        """
        if epoch < self.perceptual_warmup_epochs:
            # Linear warmup
            return self.perceptual_weight * (epoch / self.perceptual_warmup_epochs)
        return self.perceptual_weight
    
    def forward(self,
                pred_image: torch.Tensor,
                target_image: torch.Tensor,
                pred_font_attrs: Optional[torch.Tensor] = None,
                target_font_attrs: Optional[torch.Tensor] = None,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute losses
        
        Args:
            pred_image: (B, 1, H, W)
            target_image: (B, 1, H, W)
            pred_font_attrs: (B, 8) - predicted font attributes
            target_font_attrs: (B, 8) - ground truth attributes
            epoch: Current epoch (for warmup)
        
        Returns:
            Dict of losses
        """
        losses = {}
        
        # 1. Reconstruction loss (always computed)
        recon_loss = F.mse_loss(pred_image, target_image)
        losses['reconstruction'] = recon_loss
        
        # 2. Perceptual loss (computed every N batches for efficiency)
        should_compute_lpips = (self.batch_counter % self.lpips_every_n_batches == 0)
        
        if should_compute_lpips:
            # Convert grayscale to RGB for LPIPS
            pred_rgb = pred_image.repeat(1, 3, 1, 1) * 2 - 1  # [0,1] -> [-1,1]
            target_rgb = target_image.repeat(1, 3, 1, 1) * 2 - 1
            
            # Ensure on correct device
            pred_rgb = pred_rgb.to(self.device)
            target_rgb = target_rgb.to(self.device)
            
            perceptual_loss = self.lpips(pred_rgb, target_rgb).mean()
            losses['perceptual'] = perceptual_loss
        else:
            # Use zero loss when not computing (won't affect gradients)
            perceptual_loss = torch.tensor(0.0, device=pred_image.device)
            losses['perceptual'] = perceptual_loss
        
        self.batch_counter += 1
        
        # 3. Font attribute prediction (if enabled)
        if self.use_font_attrs and pred_font_attrs is not None:
            attr_loss = F.mse_loss(pred_font_attrs, target_font_attrs)
            losses['font_attr'] = attr_loss
        else:
            losses['font_attr'] = torch.tensor(0.0, device=pred_image.device)
        
        # Total loss with warmup
        perceptual_weight = self.get_perceptual_weight(epoch)
        
        total = (
            self.recon_weight * recon_loss +
            perceptual_weight * perceptual_loss
        )
        
        if losses['font_attr'].item() > 0:
            total += self.font_attr_weight * losses['font_attr']
        
        losses['total'] = total
        
        return losses


class PixelLoss(nn.Module):
    """
    Simple pixel-level reconstruction loss
    Faster alternative to LPIPS for quick experiments
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class StructuralSimilarityLoss(nn.Module):
    """
    SSIM-based loss (faster than LPIPS, better than MSE)
    Good middle ground for perceptual quality
    """
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1 - SSIM as loss
        
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        
        Returns:
            loss: scalar
        """
        # Simple SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1, padding=self.window_size // 2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim.mean()


class FastCombinedLoss(nn.Module):
    """
    Faster alternative to CombinedLoss using SSIM instead of LPIPS
    Good for quick iterations and debugging
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        loss_weights = config['training']['loss_weights']
        
        self.recon_weight = loss_weights['reconstruction']
        self.perceptual_weight = loss_weights.get('perceptual', 0.5)
        
        # Use SSIM instead of LPIPS
        self.ssim = StructuralSimilarityLoss()
        
        self.use_font_attrs = config['model']['use_font_attributes']
        if self.use_font_attrs:
            self.font_attr_weight = loss_weights['font_attr_prediction']
    
    def forward(self,
                pred_image: torch.Tensor,
                target_image: torch.Tensor,
                pred_font_attrs: Optional[torch.Tensor] = None,
                target_font_attrs: Optional[torch.Tensor] = None,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """Compute losses with SSIM"""
        losses = {}
        
        # Reconstruction
        recon_loss = F.mse_loss(pred_image, target_image)
        losses['reconstruction'] = recon_loss
        
        # Perceptual (SSIM)
        perceptual_loss = self.ssim(pred_image, target_image)
        losses['perceptual'] = perceptual_loss
        
        # Font attributes
        if self.use_font_attrs and pred_font_attrs is not None:
            attr_loss = F.mse_loss(pred_font_attrs, target_font_attrs)
            losses['font_attr'] = attr_loss
        else:
            losses['font_attr'] = torch.tensor(0.0, device=pred_image.device)
        
        # Total
        total = (
            self.recon_weight * recon_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        if losses['font_attr'].item() > 0:
            total += self.font_attr_weight * losses['font_attr']
        
        losses['total'] = total
        
        return losses


if __name__ == "__main__":
    """Test losses"""
    
    # Mock config
    config = {
        'training': {
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.5,
                'font_attr_prediction': 0.1
            },
            'lpips_every_n_batches': 4,
            'perceptual_warmup_epochs': 5
        },
        'model': {
            'use_font_attributes': True
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test data
    batch_size = 4
    pred_image = torch.randn(batch_size, 1, 128, 128).to(device)
    target_image = torch.randn(batch_size, 1, 128, 128).to(device)
    pred_attrs = torch.randn(batch_size, 8).to(device)
    target_attrs = torch.randn(batch_size, 8).to(device)
    
    # Test CombinedLoss
    print("\nTesting CombinedLoss (with LPIPS):")
    criterion = CombinedLoss(config, device=device)
    
    for epoch in [0, 3, 5, 10]:
        losses = criterion(pred_image, target_image, pred_attrs, target_attrs, epoch=epoch)
        print(f"  Epoch {epoch}:")
        print(f"    Total: {losses['total'].item():.4f}")
        print(f"    Recon: {losses['reconstruction'].item():.4f}")
        print(f"    Perceptual: {losses['perceptual'].item():.4f}")
        print(f"    Font attr: {losses['font_attr'].item():.4f}")
        print(f"    Perceptual weight: {criterion.get_perceptual_weight(epoch):.4f}")
    
    # Test FastCombinedLoss
    print("\nTesting FastCombinedLoss (with SSIM):")
    fast_criterion = FastCombinedLoss(config)
    
    losses = fast_criterion(pred_image, target_image, pred_attrs, target_attrs)
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Recon: {losses['reconstruction'].item():.4f}")
    print(f"  Perceptual (SSIM): {losses['perceptual'].item():.4f}")
    print(f"  Font attr: {losses['font_attr'].item():.4f}")
    
    print("\n✓ All loss functions working!")