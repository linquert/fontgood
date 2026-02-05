"""
Attention Analysis for Mechanistic Interpretability
Analyze attention patterns, compute entropy, detect sinks, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze attention patterns in the transformer
    
    Features:
    - Attention pattern visualization
    - Entropy computation (attention focus)
    - Attention sink detection
    - Token importance tracking
    - Head specialization analysis
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def get_attention_patterns(self,
                              images: torch.Tensor,
                              char_indices: torch.Tensor,
                              font_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention patterns for all layers
        
        Returns:
            List of attention tensors: (B, H, N, N) for each layer
        """
        output = self.model(
            image=images.to(self.device),
            char_indices=char_indices.to(self.device),
            font_idx=font_idx.to(self.device),
            return_attention=True
        )
        
        return output['attention']
    
    def compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention distribution
        
        Higher entropy = more diffuse attention
        Lower entropy = more focused attention
        
        Args:
            attention: (B, H, N, N)
        
        Returns:
            entropy: (B, H, N) - entropy for each query token
        """
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(
            attention * torch.log(attention + 1e-10),
            dim=-1
        )
        
        return entropy
    
    def detect_attention_sinks(self,
                              attention: torch.Tensor,
                              threshold: float = 0.3) -> Dict:
        """
        Detect 'attention sinks' - tokens that receive high attention
        
        Args:
            attention: (B, H, N, N)
            threshold: Minimum average attention to be considered a sink
        
        Returns:
            Dict with sink analysis
        """
        # Average attention received by each token across all queries
        # Sum over query dimension, average over batch and heads
        attention_received = attention.mean(dim=(0, 1, 2))  # (N,)
        
        sinks = (attention_received > threshold).nonzero(as_tuple=True)[0]
        
        return {
            'sink_positions': sinks.cpu().tolist(),
            'sink_weights': attention_received[sinks].cpu().tolist(),
            'attention_received': attention_received.cpu()
        }
    
    def analyze_token_importance(self,
                                 attention_patterns: List[torch.Tensor],
                                 num_special_tokens: int = 3) -> Dict:
        """
        Analyze which tokens are most important across layers
        
        Args:
            attention_patterns: List of (B, H, N, N) tensors
            num_special_tokens: Font + char tokens
        
        Returns:
            Dict with importance scores
        """
        all_importance = []
        
        for layer_attn in attention_patterns:
            # Average attention received by each token
            importance = layer_attn.mean(dim=(0, 1, 2))  # (N,)
            all_importance.append(importance.cpu())
        
        all_importance = torch.stack(all_importance)  # (L, N)
        
        # Separate special tokens vs spatial tokens
        special_importance = all_importance[:, :num_special_tokens]
        spatial_importance = all_importance[:, num_special_tokens:]
        
        return {
            'special_tokens': special_importance,  # (L, num_special)
            'spatial_tokens': spatial_importance,  # (L, num_spatial)
            'font_token': all_importance[:, 0],  # (L,)
            'char_tokens': all_importance[:, 1:num_special_tokens],  # (L, N_chars)
        }
    
    def analyze_head_specialization(self,
                                   attention_patterns: List[torch.Tensor],
                                   num_special_tokens: int = 3) -> Dict:
        """
        Analyze if different heads specialize in different patterns
        
        Returns:
            Dict with per-head statistics
        """
        num_layers = len(attention_patterns)
        num_heads = attention_patterns[0].shape[1]
        
        head_stats = []
        
        for layer_idx, layer_attn in enumerate(attention_patterns):
            # (B, H, N, N)
            
            for head_idx in range(num_heads):
                head_attn = layer_attn[:, head_idx, :, :]  # (B, N, N)
                
                # Attention to special tokens
                special_attn = head_attn[:, :, :num_special_tokens].mean().item()
                
                # Attention to spatial tokens
                spatial_attn = head_attn[:, :, num_special_tokens:].mean().item()
                
                # Attention entropy
                entropy = self.compute_attention_entropy(
                    layer_attn[:, head_idx:head_idx+1, :, :]
                ).mean().item()
                
                head_stats.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'special_attn': special_attn,
                    'spatial_attn': spatial_attn,
                    'entropy': entropy
                })
        
        return head_stats
    
    def visualize_attention_map(self,
                               attention: torch.Tensor,
                               layer_idx: int,
                               head_idx: int,
                               sample_idx: int = 0,
                               save_path: Optional[str] = None,
                               token_labels: Optional[List[str]] = None):
        """
        Visualize attention pattern for a specific layer/head
        
        Args:
            attention: (B, H, N, N)
            layer_idx: Which layer
            head_idx: Which head
            sample_idx: Which sample in batch
            save_path: Where to save figure
            token_labels: Optional labels for tokens
        """
        attn_map = attention[sample_idx, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn_map,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )
        
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
        
        if token_labels is not None:
            plt.xticks(np.arange(len(token_labels)) + 0.5, token_labels, rotation=90)
            plt.yticks(np.arange(len(token_labels)) + 0.5, token_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention map to {save_path}")
        
        plt.close()
    
    def visualize_attention_flow(self,
                                attention_patterns: List[torch.Tensor],
                                sample_idx: int = 0,
                                head_idx: int = 0,
                                save_path: Optional[str] = None):
        """
        Visualize how attention changes across layers
        
        Shows attention from first char token to other positions
        """
        num_layers = len(attention_patterns)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for layer_idx in range(min(8, num_layers)):
            attn = attention_patterns[layer_idx][sample_idx, head_idx].cpu().numpy()
            
            # Attention from position 1 (first char token)
            char_attn = attn[1, :]
            
            axes[layer_idx].bar(range(len(char_attn)), char_attn)
            axes[layer_idx].set_title(f'Layer {layer_idx}')
            axes[layer_idx].set_xlabel('Token Position')
            axes[layer_idx].set_ylabel('Attention')
            
            # Mark special token region
            axes[layer_idx].axvline(x=2.5, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Attention from First Char Token Across Layers')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention flow to {save_path}")
        
        plt.close()
    
    def compute_layer_statistics(self,
                                attention_patterns: List[torch.Tensor]) -> Dict:
        """
        Compute aggregate statistics for each layer
        """
        stats = []
        
        for layer_idx, attn in enumerate(attention_patterns):
            # (B, H, N, N)
            
            # Average entropy per head
            entropy = self.compute_attention_entropy(attn)
            avg_entropy = entropy.mean(dim=(0, 2))  # (H,)
            
            # Attention to first token (potential sink)
            first_token_attn = attn[:, :, :, 0].mean()
            
            # Sparsity (% of weights below threshold)
            sparsity = (attn < 0.01).float().mean()
            
            stats.append({
                'layer': layer_idx,
                'entropy_per_head': avg_entropy.cpu().tolist(),
                'avg_entropy': avg_entropy.mean().item(),
                'first_token_attn': first_token_attn.item(),
                'sparsity': sparsity.item()
            })
        
        return stats


if __name__ == "__main__":
    print("Testing Attention Analyzer")
    print("=" * 50)
    
    # Create dummy attention patterns
    batch_size = 4
    num_heads = 8
    seq_len = 67  # 1 font + 2 chars + 64 patches
    num_layers = 8
    
    attention_patterns = []
    for _ in range(num_layers):
        # Random attention (should sum to 1 in last dim)
        attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attention_patterns.append(attn)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)
        
        def forward(self, image, char_indices, font_idx, return_attention=False):
            if return_attention:
                return {'attention': attention_patterns}
            return {}
    
    model = DummyModel()
    analyzer = AttentionAnalyzer(model)
    
    # Test entropy
    entropy = analyzer.compute_attention_entropy(attention_patterns[0])
    print(f"✓ Attention entropy: {entropy.shape}")
    
    # Test attention sinks
    sinks = analyzer.detect_attention_sinks(attention_patterns[0])
    print(f"✓ Attention sinks detected: {len(sinks['sink_positions'])}")
    
    # Test token importance
    importance = analyzer.analyze_token_importance(attention_patterns, num_special_tokens=3)
    print(f"✓ Token importance:")
    print(f"  Font token across layers: {importance['font_token'].shape}")
    
    # Test head specialization
    head_stats = analyzer.analyze_head_specialization(attention_patterns)
    print(f"✓ Head statistics computed: {len(head_stats)} head-layer pairs")
    
    # Test layer statistics
    layer_stats = analyzer.compute_layer_statistics(attention_patterns)
    print(f"✓ Layer statistics: {len(layer_stats)} layers")
    
    print("\n✓ All attention analysis tests passed!")
