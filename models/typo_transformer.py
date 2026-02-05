"""
Pure Typographic Transformer for Mechanistic Interpretability

Architecture:
[font_tok, char_tok_1, ..., char_tok_N, patch_1, ..., patch_64]
         ↓
   Pure Transformer (8 layers, self-attention only)
         ↓
   Spatial Decoder (uses all patch tokens)
         ↓
   Reconstructed Image

NO VAE, NO cross-attention, NO information bottlenecks
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .transformer import TransformerBlock, RotaryEmbedding, RMSNorm
from .image_coding import PatchEmbedding, SpatialDecoder


class TypographicTransformer(nn.Module):
    """
    Pure transformer for typography with maximum interpretability
    
    Key features:
    - Prepended special tokens (font + characters)
    - Uniform self-attention architecture
    - No information bottlenecks
    - Activation hooks at every layer
    - Clear token attribution
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.max_seq_len = config['model']['max_seq_len']
        
        # Extract dimensions
        self.char_vocab_size = config['model']['char_vocab_size']
        self.char_embed_dim = config['model']['char_embed_dim']
        self.font_embed_dim = config['model']['font_embed_dim']
        self.num_fonts = config['model']['num_fonts']
        self.transformer_dim = config['model']['transformer']['dim']
        
        # 1. Character Embeddings (learned)
        self.char_embeddings = nn.Embedding(
            self.char_vocab_size,
            self.char_embed_dim
        )
        
        # Project character embeddings to transformer dimension
        self.char_proj = nn.Linear(self.char_embed_dim, self.transformer_dim, bias=False)
        
        # 2. Font Embeddings (learned)
        self.font_embeddings = nn.Embedding(
            self.num_fonts,
            self.font_embed_dim
        )
        
        # Project font embeddings to transformer dimension
        self.font_proj = nn.Linear(self.font_embed_dim, self.transformer_dim, bias=False)
        
        # Optional: Font attribute prediction (auxiliary task)
        if config['model']['use_font_attributes']:
            self.font_attr_predictor = nn.Linear(
                self.font_embed_dim,
                config['model']['font_attr_dim']
            )
        else:
            self.font_attr_predictor = None
        
        # 3. Image Patch Embeddings
        patch_config = config['model']['patch_encoder']
        self.patch_encoder = PatchEmbedding(
            image_size=config['data']['image_size'],
            patch_size=patch_config['patch_size'],
            in_channels=patch_config['in_channels'],
            embed_dim=self.transformer_dim
        )
        
        # 4. Rotary Embeddings
        transformer_config = config['model']['transformer']
        if transformer_config['use_rotary_emb']:
            self.rope = RotaryEmbedding(
                dim=self.transformer_dim // transformer_config['num_heads'],
                max_seq_len=512
            )
        else:
            self.rope = None
        
        # 5. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.transformer_dim,
                num_heads=transformer_config['num_heads'],
                mlp_ratio=transformer_config['mlp_ratio'],
                dropout=transformer_config['dropout']
            )
            for _ in range(transformer_config['num_layers'])
        ])
        
        # 6. Final norm
        self.final_norm = RMSNorm(self.transformer_dim)
        
        # 7. Spatial Decoder
        decoder_config = config['model']['decoder']
        self.decoder = SpatialDecoder(
            embed_dim=self.transformer_dim,
            grid_size=self.patch_encoder.get_patch_grid_size(),
            hidden_dims=decoder_config['hidden_dims'],
            output_channels=decoder_config['output_channels']
        )
        
        # Storage for interpretability
        self.all_layer_outputs = []
        self.all_attention_patterns = []
        
    def encode_font(self, font_idx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode font index to embedding
        
        Args:
            font_idx: (B,)
        
        Returns:
            font_token: (B, 1, D)
            font_attrs: (B, attr_dim) if predictor enabled
        """
        # Get font embedding
        font_emb = self.font_embeddings(font_idx)  # (B, font_embed_dim)
        
        # Predict attributes if enabled (auxiliary task)
        font_attrs = None
        if self.font_attr_predictor is not None:
            font_attrs = self.font_attr_predictor(font_emb)  # (B, attr_dim)
        
        # Project to transformer dimension
        font_token = self.font_proj(font_emb)  # (B, D)
        font_token = font_token.unsqueeze(1)  # (B, 1, D)
        
        return font_token, font_attrs
    
    def encode_characters(self, char_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode character sequence
        
        Args:
            char_indices: (B, N) where N is 1-3
        
        Returns:
            char_tokens: (B, N, D)
        """
        # Get character embeddings
        char_emb = self.char_embeddings(char_indices)  # (B, N, char_embed_dim)
        
        # Project to transformer dimension
        char_tokens = self.char_proj(char_emb)  # (B, N, D)
        
        return char_tokens
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to patch tokens
        
        Args:
            image: (B, 1, H, W)
        
        Returns:
            patch_tokens: (B, num_patches, D)
        """
        return self.patch_encoder(image)
    
    def forward(self,
                image: torch.Tensor,
                char_indices: torch.Tensor,
                font_idx: torch.Tensor,
                return_attention: bool = False,
                return_activations: bool = False,
                return_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            image: (B, 1, H, W)
            char_indices: (B, N) - sequence of 1-3 characters
            font_idx: (B,) - font index
            return_attention: Return attention patterns
            return_activations: Return layer activations
            return_tokens: Return final token representations
        
        Returns:
            Dict with:
                - reconstructed: (B, 1, H, W)
                - font_attrs_pred: (B, attr_dim) if enabled
                - attention: List of (B, H, N, N) if requested
                - activations: List of dicts if requested
                - tokens: (B, total_seq_len, D) if requested
        """
        B = image.shape[0]
        
        # 1. Encode all inputs
        font_token, font_attrs_pred = self.encode_font(font_idx)
        char_tokens = self.encode_characters(char_indices)
        patch_tokens = self.encode_image(image)
        
        # 2. Concatenate all tokens
        # [font, char_1, ..., char_N, patch_1, ..., patch_64]
        tokens = torch.cat([font_token, char_tokens, patch_tokens], dim=1)
        
        seq_len = tokens.shape[1]
        num_special_tokens = 1 + char_indices.shape[1]  # font + chars
        
        # 3. Get RoPE embeddings
        if self.rope is not None:
            rope = self.rope(seq_len)
        else:
            rope = None
        
        # 4. Pass through transformer
        self.all_layer_outputs = []
        self.all_attention_patterns = []
        
        for block in self.blocks:
            tokens, attn = block(
                tokens,
                rope=rope,
                return_attention=return_attention
            )
            
            if return_activations:
                self.all_layer_outputs.append(block.get_activations())
            
            if return_attention and attn is not None:
                self.all_attention_patterns.append(attn)
        
        # 5. Final norm
        tokens = self.final_norm(tokens)
        
        # 6. Extract spatial tokens for decoding
        # Skip special tokens (font + characters)
        spatial_tokens = tokens[:, num_special_tokens:, :]
        
        # 7. Decode to image
        reconstructed = self.decoder(spatial_tokens)
        
        # 8. Prepare output
        output = {
            'reconstructed': reconstructed,
        }
        
        if font_attrs_pred is not None:
            output['font_attrs_pred'] = font_attrs_pred
        
        if return_attention:
            output['attention'] = self.all_attention_patterns
        
        if return_activations:
            output['activations'] = self.all_layer_outputs
        
        if return_tokens:
            output['tokens'] = tokens
            output['font_token'] = tokens[:, 0, :]
            output['char_tokens'] = tokens[:, 1:num_special_tokens, :]
            output['spatial_tokens'] = spatial_tokens
        
        return output
    
    def interpolate_fonts(self,
                         char_indices: torch.Tensor,
                         font_idx1: torch.Tensor,
                         font_idx2: torch.Tensor,
                         num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two fonts in embedding space
        
        Args:
            char_indices: (B, N)
            font_idx1: (B,) - start font
            font_idx2: (B,) - end font
            num_steps: interpolation steps
        
        Returns:
            images: (B, num_steps, 1, H, W)
        """
        B = char_indices.shape[0]
        device = char_indices.device
        
        # Get font embeddings
        font_emb1 = self.font_embeddings(font_idx1)
        font_emb2 = self.font_embeddings(font_idx2)
        
        # Encode characters
        char_tokens = self.encode_characters(char_indices)
        
        # Create dummy image for patch tokens
        dummy_image = torch.zeros(B, 1, 128, 128).to(device)
        patch_tokens = self.encode_image(dummy_image)
        
        images = []
        
        for alpha in torch.linspace(0, 1, num_steps):
            # Interpolate font embedding
            font_emb_interp = (1 - alpha) * font_emb1 + alpha * font_emb2
            font_token_interp = self.font_proj(font_emb_interp).unsqueeze(1)
            
            # Build token sequence
            tokens = torch.cat([font_token_interp, char_tokens, patch_tokens], dim=1)
            
            # Get RoPE
            if self.rope is not None:
                rope = self.rope(tokens.shape[1])
            else:
                rope = None
            
            # Transform
            for block in self.blocks:
                tokens, _ = block(tokens, rope=rope)
            
            tokens = self.final_norm(tokens)
            
            # Decode
            num_special = 1 + char_indices.shape[1]
            spatial_tokens = tokens[:, num_special:, :]
            img = self.decoder(spatial_tokens)
            
            images.append(img)
        
        return torch.stack(images, dim=1)


if __name__ == "__main__":
    import yaml
    
    # Create minimal config for testing
    config = {
        'data': {'image_size': 128},
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
                'embed_dim': 512
            },
            'transformer': {
                'num_layers': 8,
                'num_heads': 8,
                'dim': 512,
                'mlp_ratio': 4,
                'dropout': 0.1,
                'use_rmsnorm': True,
                'use_swiglu': True,
                'use_rotary_emb': True,
            },
            'decoder': {
                'type': 'spatial_cnn',
                'hidden_dims': [256, 128, 64],
                'output_channels': 1
            }
        }
    }
    
    print("Testing Pure Typographic Transformer")
    print("=" * 50)
    
    # Create model
    model = TypographicTransformer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params/1e6:.2f}M parameters")
    
    # Test forward pass
    batch_size = 4
    image = torch.randn(batch_size, 1, 128, 128)
    char_indices = torch.randint(0, 52, (batch_size, 2))  # 2-char sequence
    font_idx = torch.randint(0, 500, (batch_size,))
    
    output = model(
        image=image,
        char_indices=char_indices,
        font_idx=font_idx,
        return_attention=True,
        return_activations=True,
        return_tokens=True
    )
    
    print(f"\n✓ Forward pass:")
    print(f"  Input image: {image.shape}")
    print(f"  Char indices: {char_indices.shape}")
    print(f"  Font idx: {font_idx.shape}")
    print(f"\n  Output:")
    print(f"    Reconstructed: {output['reconstructed'].shape}")
    print(f"    Font attrs pred: {output['font_attrs_pred'].shape}")
    print(f"    Num attention layers: {len(output['attention'])}")
    print(f"    Num activation layers: {len(output['activations'])}")
    print(f"    Tokens shape: {output['tokens'].shape}")
    print(f"    Font token: {output['font_token'].shape}")
    print(f"    Char tokens: {output['char_tokens'].shape}")
    print(f"    Spatial tokens: {output['spatial_tokens'].shape}")
    
    # Test interpolation
    font_idx1 = torch.tensor([0])
    font_idx2 = torch.tensor([1])
    char_test = torch.tensor([[0]])  # 'A'
    
    interpolated = model.interpolate_fonts(char_test, font_idx1, font_idx2, num_steps=5)
    print(f"\n✓ Font interpolation: {interpolated.shape}")
    
    print("\n✓ Model working perfectly!")
