"""
Pure Transformer Components for Mechanistic Interpretability

Design decisions:
- RMSNorm (simpler than LayerNorm)
- SwiGLU (proven better than GELU)
- RoPE (interpretable positional encoding)
- Pre-norm (better gradient flow)
- No cross-attention (uniform architecture)
- Activation hooks everywhere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Normalization - simpler than LayerNorm"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to queries and keys"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation - better than GELU for transformers"""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with comprehensive hooks
    
    CRITICAL for mech interp:
    - Separate Q, K, V projections (no bias)
    - Store attention patterns per head
    - Store value outputs per head (before mixing)
    - Clear intervention points
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections (no bias for cleaner interpretation)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Cached for interpretability
        self.attention_weights = None  # (B, H, N, N)
        self.value_mix = None          # (B, H, N, D_head) before output projection
        self.queries = None
        self.keys = None
        self.values = None
    
    def forward(self,
                x: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, D)
            rope: Optional (cos, sin) for rotary embeddings
        
        Returns:
            output: (B, N, D)
            attention: (B, H, N, N) if return_attention
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Store for interpretability
        self.queries = q.detach()
        self.keys = k.detach()
        self.values = v.detach()
        
        # Apply RoPE if provided
        if rope is not None:
            cos, sin = rope
            cos = cos[:N].unsqueeze(0).unsqueeze(0)
            sin = sin[:N].unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Store attention patterns
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Store value mix before output projection
        self.value_mix = out.detach()
        
        # Output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.o_proj(out)
        
        if return_attention:
            return out, attn
        return out, None
    
    def get_attention_entropy(self) -> torch.Tensor:
        """Compute entropy of attention distribution (measure of focus)"""
        if self.attention_weights is None:
            return None
        attn = self.attention_weights
        entropy = -torch.sum(attn * torch.log(attn + 1e-9), dim=-1)
        return entropy.mean(dim=(0, 2))  # (H,) - entropy per head


class TransformerBlock(nn.Module):
    """
    Transformer block with clear residual stream
    
    x = x + attn(norm(x))
    x = x + mlp(norm(x))
    
    This makes information flow completely transparent
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Pre-norm architecture
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim=dim * mlp_ratio)
        
        # Activation storage for interpretability
        self.activations = {}
    
    def forward(self,
                x: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, D)
            rope: RoPE embeddings
        
        Returns:
            x: (B, N, D)
            attention: (B, H, N, N) if return_attention
        """
        # Store pre-attention state
        self.activations['input'] = x.detach()
        
        # Self-attention with residual
        attn_out, attn_weights = self.attn(
            self.norm1(x),
            rope=rope,
            return_attention=return_attention
        )
        
        self.activations['post_attn_pre_residual'] = attn_out.detach()
        x = x + attn_out
        self.activations['post_attn'] = x.detach()
        
        # MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        self.activations['post_mlp_pre_residual'] = mlp_out.detach()
        x = x + mlp_out
        self.activations['post_mlp'] = x.detach()
        
        return x, attn_weights
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return all stored activations"""
        return self.activations
    
    def clear_activations(self):
        """Clear activation cache"""
        self.activations = {}


if __name__ == "__main__":
    # Test components
    batch_size, seq_len, dim = 2, 16, 512
    num_heads = 8
    
    print("Testing Pure Transformer Components")
    print("=" * 50)
    
    # Test RMSNorm
    norm = RMSNorm(dim)
    x = torch.randn(batch_size, seq_len, dim)
    x_normed = norm(x)
    print(f"✓ RMSNorm: {x.shape} -> {x_normed.shape}")
    print(f"  Mean: {x_normed.mean():.4f}, Std: {x_normed.std():.4f}")
    
    # Test RoPE
    rope = RotaryEmbedding(dim // num_heads)
    cos, sin = rope(seq_len)
    print(f"✓ RoPE: cos={cos.shape}, sin={sin.shape}")
    
    # Test SwiGLU
    mlp = SwiGLU(dim)
    mlp_out = mlp(x)
    print(f"✓ SwiGLU: {x.shape} -> {mlp_out.shape}")
    
    # Test Attention
    attn = MultiHeadSelfAttention(dim, num_heads)
    attn_out, attn_weights = attn(x, rope=(cos, sin), return_attention=True)
    print(f"✓ Attention: {x.shape} -> {attn_out.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    print(f"  Stored queries: {attn.queries.shape}")
    
    entropy = attn.get_attention_entropy()
    print(f"  Attention entropy per head: {entropy}")
    
    # Test TransformerBlock
    block = TransformerBlock(dim, num_heads)
    block_out, block_attn = block(x, rope=(cos, sin), return_attention=True)
    print(f"✓ TransformerBlock: {x.shape} -> {block_out.shape}")
    
    acts = block.get_activations()
    print(f"  Stored activations: {list(acts.keys())}")
    
    print("\n✓ All components working correctly!")
