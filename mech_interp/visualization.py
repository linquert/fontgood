"""
Visualization Tools for Mechanistic Interpretability
Comprehensive plotting for probes, attention, activations, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


class InterpVisualizer:
    """
    Comprehensive visualization for mechanistic interpretability
    """
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_probe_accuracies(self,
                             probe_results: Dict,
                             probe_names: Optional[List[str]] = None,
                             save_name: str = 'probe_accuracies.png'):
        """
        Plot probe accuracies across layers
        
        Args:
            probe_results: Dict mapping (layer, probe_name) -> metrics
            probe_names: Which probes to plot (None = all)
        """
        # Organize data
        data = {}
        for (layer, probe_name), metrics in probe_results.items():
            if probe_names is None or probe_name in probe_names:
                if probe_name not in data:
                    data[probe_name] = {}
                data[probe_name][layer] = metrics['accuracy']
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for probe_name, layer_accs in data.items():
            layers = sorted(layer_accs.keys())
            accs = [layer_accs[l] for l in layers]
            
            ax.plot(layers, accs, marker='o', linewidth=2, label=probe_name)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Linear Probe Accuracies Across Layers', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved probe accuracies to {save_path}")
        plt.close()
    
    def plot_attention_entropy(self,
                              layer_stats: List[Dict],
                              save_name: str = 'attention_entropy.png'):
        """
        Plot attention entropy across layers and heads
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data
        layers = [s['layer'] for s in layer_stats]
        avg_entropy = [s['avg_entropy'] for s in layer_stats]
        
        # Plot 1: Average entropy per layer
        ax1.plot(layers, avg_entropy, marker='o', linewidth=2, color='steelblue')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Average Attention Entropy', fontsize=12)
        ax1.set_title('Attention Entropy by Layer', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Entropy per head (heatmap)
        entropy_matrix = []
        for s in layer_stats:
            entropy_matrix.append(s['entropy_per_head'])
        
        entropy_matrix = np.array(entropy_matrix)
        
        im = ax2.imshow(entropy_matrix.T, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Head', fontsize=12)
        ax2.set_title('Entropy per Head', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Entropy')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved attention entropy to {save_path}")
        plt.close()
    
    def plot_token_importance(self,
                             importance_dict: Dict,
                             save_name: str = 'token_importance.png'):
        """
        Plot token importance across layers
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Font token
        font_importance = importance_dict['font_token']
        axes[0].plot(range(len(font_importance)), font_importance, 
                    marker='o', linewidth=2, color='coral')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Importance')
        axes[0].set_title('Font Token Importance', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Character tokens
        char_importance = importance_dict['char_tokens']  # (L, N_chars)
        for i in range(min(3, char_importance.shape[1])):
            axes[1].plot(range(len(char_importance)), char_importance[:, i],
                        marker='o', linewidth=2, label=f'Char {i+1}')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Importance')
        axes[1].set_title('Character Token Importance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Spatial tokens (average)
        spatial_importance = importance_dict['spatial_tokens']  # (L, N_spatial)
        spatial_avg = spatial_importance.mean(dim=1)
        spatial_std = spatial_importance.std(dim=1)
        
        axes[2].plot(range(len(spatial_avg)), spatial_avg, 
                    marker='o', linewidth=2, color='steelblue')
        axes[2].fill_between(range(len(spatial_avg)),
                            spatial_avg - spatial_std,
                            spatial_avg + spatial_std,
                            alpha=0.3, color='steelblue')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Average Importance')
        axes[2].set_title('Spatial Tokens (Mean ± Std)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Special vs Spatial
        special_avg = importance_dict['special_tokens'].mean(dim=1)
        axes[3].plot(range(len(special_avg)), special_avg, 
                    marker='o', linewidth=2, label='Special Tokens', color='coral')
        axes[3].plot(range(len(spatial_avg)), spatial_avg,
                    marker='s', linewidth=2, label='Spatial Tokens', color='steelblue')
        axes[3].set_xlabel('Layer')
        axes[3].set_ylabel('Average Importance')
        axes[3].set_title('Special vs Spatial Tokens', fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved token importance to {save_path}")
        plt.close()
    
    def plot_reconstruction_samples(self,
                                   originals: torch.Tensor,
                                   reconstructed: torch.Tensor,
                                   sequences: List[str],
                                   num_samples: int = 8,
                                   save_name: str = 'reconstructions.png'):
        """
        Plot original vs reconstructed images
        """
        num_samples = min(num_samples, originals.shape[0])
        
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(originals[i, 0].cpu(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontweight='bold')
            axes[0, i].text(0.5, -0.1, sequences[i], 
                          transform=axes[0, i].transAxes,
                          ha='center', fontsize=10)
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i, 0].cpu(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved reconstructions to {save_path}")
        plt.close()
    
    def plot_font_interpolation(self,
                               interpolated: torch.Tensor,
                               font1_name: str,
                               font2_name: str,
                               char: str,
                               save_name: str = 'font_interpolation.png'):
        """
        Plot font interpolation results
        
        Args:
            interpolated: (1, num_steps, 1, H, W)
        """
        num_steps = interpolated.shape[1]
        
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        
        for i in range(num_steps):
            axes[i].imshow(interpolated[0, i, 0].cpu(), cmap='gray')
            axes[i].axis('off')
            alpha = i / (num_steps - 1)
            axes[i].set_title(f'α={alpha:.2f}', fontsize=10)
        
        fig.suptitle(f"'{char}': {font1_name} → {font2_name}", 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved interpolation to {save_path}")
        plt.close()
    
    def plot_training_curves(self,
                            train_losses: List[float],
                            val_losses: List[float],
                            save_name: str = 'training_curves.png'):
        """Plot training and validation loss curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(len(train_losses))
        
        ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='steelblue')
        ax.plot(epochs, val_losses, label='Val Loss', linewidth=2, color='coral')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        plt.close()
    
    def plot_head_specialization(self,
                                head_stats: List[Dict],
                                save_name: str = 'head_specialization.png'):
        """
        Plot head specialization patterns
        """
        # Organize by layer and head
        num_layers = max(s['layer'] for s in head_stats) + 1
        num_heads = max(s['head'] for s in head_stats) + 1
        
        special_attn = np.zeros((num_layers, num_heads))
        spatial_attn = np.zeros((num_layers, num_heads))
        
        for s in head_stats:
            special_attn[s['layer'], s['head']] = s['special_attn']
            spatial_attn[s['layer'], s['head']] = s['spatial_attn']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Special token attention
        im1 = ax1.imshow(special_attn.T, aspect='auto', cmap='Reds')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Head', fontsize=12)
        ax1.set_title('Attention to Special Tokens', fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax1)
        
        # Spatial token attention
        im2 = ax2.imshow(spatial_attn.T, aspect='auto', cmap='Blues')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Head', fontsize=12)
        ax2.set_title('Attention to Spatial Tokens', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved head specialization to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Testing Visualization Tools")
    print("=" * 50)
    
    viz = InterpVisualizer('./test_viz')
    
    # Test probe accuracies
    probe_results = {
        (0, 'char_id'): {'accuracy': 0.5},
        (2, 'char_id'): {'accuracy': 0.85},
        (4, 'char_id'): {'accuracy': 0.95},
        (0, 'serif'): {'accuracy': 0.65},
        (2, 'serif'): {'accuracy': 0.85},
        (4, 'serif'): {'accuracy': 0.92},
    }
    
    viz.plot_probe_accuracies(probe_results)
    print("✓ Plotted probe accuracies")
    
    # Test reconstruction samples
    originals = torch.rand(8, 1, 128, 128)
    reconstructed = torch.rand(8, 1, 128, 128)
    sequences = ['A', 'B', 'AB', 'the', 'Ty', 'We', 'XYZ', 'abc']
    
    viz.plot_reconstruction_samples(originals, reconstructed, sequences)
    print("✓ Plotted reconstructions")
    
    # Test interpolation
    interpolated = torch.rand(1, 10, 1, 128, 128)
    viz.plot_font_interpolation(interpolated, 'Arial', 'Times', 'A')
    print("✓ Plotted interpolation")
    
    print("\n✓ All visualization tests passed!")
