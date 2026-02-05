"""
Image Encoding and Decoding for Pure Transformer Architecture

Key decisions:
- Patch embeddings (not CNN) for uniform architecture
- Spatial decoder using all tokens (no pooling)
- No information bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Convert image to patch tokens
    
    For 128x128 image with 16x16 patches:
    - 64 patches (8x8 grid)
    - Each patch is a token
    - Clean, uniform tokenization
    """
    
    def __init__(self,
                 image_size: int = 128,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 embed_dim: int = 512):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        
        # Single conv layer for patch projection
        # This is cleaner than multi-layer CNN
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False  # No bias for cleaner interpretation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x
    
    def get_patch_grid_size(self) -> int:
        """Return grid size for spatial decoder"""
        return self.grid_size


class SpatialDecoder(nn.Module):
    """
    Decode from spatial tokens to image
    
    CRITICAL: Uses ALL spatial tokens, no pooling
    This preserves spatial information for interpretability
    """
    
    def __init__(self,
                 embed_dim: int = 512,
                 grid_size: int = 8,
                 hidden_dims: list = [256, 128, 64],
                 output_channels: int = 1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Build decoder layers
        layers = []
        in_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(
                    in_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        
        # Final layer to output channels
        layers.append(
            nn.ConvTranspose2d(
                in_dim,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True
            )
        )
        
        self.decoder = nn.Sequential(*layers)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, num_patches, embed_dim)
        
        Returns:
            image: (B, 1, H, W)
        """
        B = tokens.shape[0]
        
        # Reshape tokens to spatial grid
        # (B, num_patches, D) -> (B, D, grid_size, grid_size)
        x = rearrange(
            tokens,
            'b (h w) d -> b d h w',
            h=self.grid_size,
            w=self.grid_size
        )
        
        # Apply decoder
        x = self.decoder(x)
        
        # Apply output activation
        x = self.output_activation(x)
        
        return x


class PositionEmbedding(nn.Module):
    """
    Optional learned position embeddings for tokens
    (Alternative to RoPE for non-attention parts)
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Return position embeddings for sequence length"""
        return self.pos_emb[:, :seq_len, :]


if __name__ == "__main__":
    print("Testing Image Encoding/Decoding")
    print("=" * 50)
    
    batch_size = 4
    image_size = 128
    patch_size = 16
    embed_dim = 512
    
    # Test Patch Embedding
    patch_emb = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=1,
        embed_dim=embed_dim
    )
    
    x = torch.randn(batch_size, 1, image_size, image_size)
    patches = patch_emb(x)
    
    print(f"✓ Patch Embedding:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {patches.shape}")
    print(f"  Grid size: {patch_emb.get_patch_grid_size()}")
    print(f"  Num patches: {patch_emb.num_patches}")
    
    # Test Spatial Decoder
    decoder = SpatialDecoder(
        embed_dim=embed_dim,
        grid_size=patch_emb.get_patch_grid_size(),
        hidden_dims=[256, 128, 64],
        output_channels=1
    )
    
    reconstructed = decoder(patches)
    
    print(f"\n✓ Spatial Decoder:")
    print(f"  Input: {patches.shape}")
    print(f"  Output: {reconstructed.shape}")
    print(f"  Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Test end-to-end
    print(f"\n✓ End-to-end reconstruction:")
    print(f"  Original: {x.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  MSE: {F.mse_loss(x.sigmoid(), reconstructed):.4f}")
    
    # Test position embeddings
    pos_emb = PositionEmbedding(max_seq_len=100, embed_dim=embed_dim)
    pos = pos_emb(seq_len=patches.shape[1])
    print(f"\n✓ Position Embeddings:")
    print(f"  Shape: {pos.shape}")
    
    print("\n✓ All components working!")
