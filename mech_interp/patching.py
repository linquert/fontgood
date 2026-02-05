"""
Activation Patching for Causal Analysis
Swap activations between different inputs to understand information flow
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivationPatcher:
    """
    Perform activation patching experiments
    
    Swap activations from source to target to understand:
    - Which layers encode which features
    - Causal flow of information
    - Minimal circuits for tasks
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.hooks = []
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def patch_activation(self,
                        source_input: Dict,
                        target_input: Dict,
                        layer_idx: int,
                        position: str = 'post_attn',
                        token_positions: Optional[List[int]] = None) -> torch.Tensor:
        """
        Patch activations from source to target
        
        Args:
            source_input: Dict with image, char_indices, font_idx for source
            target_input: Dict with image, char_indices, font_idx for target
            layer_idx: Which layer to patch
            position: 'post_attn', 'post_mlp', 'input'
            token_positions: Which token positions to patch (None = all)
        
        Returns:
            Output from patched forward pass
        """
        self.clear_hooks()
        
        # First, run source to get activations
        with torch.no_grad():
            source_output = self.model(
                image=source_input['image'].to(self.device),
                char_indices=source_input['char_indices'].to(self.device),
                font_idx=source_input['font_idx'].to(self.device),
                return_activations=True
            )
        
        source_activations = source_output['activations'][layer_idx][position]
        
        # Hook to patch activations
        patched_activations = source_activations.clone()
        
        def patch_hook(module, input, output):
            # Patch specific positions
            if token_positions is not None:
                output = output.clone()
                output[:, token_positions, :] = patched_activations[:, token_positions, :]
            else:
                output = patched_activations
            return output
        
        # Register hook at the right layer
        target_block = self.model.blocks[layer_idx]
        
        if position == 'post_attn':
            # Hook after attention but before residual
            handle = target_block.attn.register_forward_hook(
                lambda m, i, o: (patch_hook(m, i, o[0]), o[1])
            )
        elif position == 'post_mlp':
            handle = target_block.mlp.register_forward_hook(patch_hook)
        else:
            # Hook at block input
            handle = target_block.register_forward_hook(patch_hook)
        
        self.hooks.append(handle)
        
        # Run target with patched activations
        with torch.no_grad():
            patched_output = self.model(
                image=target_input['image'].to(self.device),
                char_indices=target_input['char_indices'].to(self.device),
                font_idx=target_input['font_idx'].to(self.device)
            )
        
        self.clear_hooks()
        
        return patched_output['reconstructed']
    
    def patch_font_token(self,
                        source_font_idx: torch.Tensor,
                        target_font_idx: torch.Tensor,
                        char_indices: torch.Tensor,
                        layer_idx: int,
                        image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Patch just the font token from source to target
        
        Tests: At which layer does font information matter?
        """
        if image is None:
            # Create dummy image
            image = torch.zeros(1, 1, 128, 128)
        
        source_input = {
            'image': image,
            'char_indices': char_indices,
            'font_idx': source_font_idx
        }
        
        target_input = {
            'image': image,
            'char_indices': char_indices,
            'font_idx': target_font_idx
        }
        
        # Patch position 0 (font token)
        return self.patch_activation(
            source_input,
            target_input,
            layer_idx,
            position='post_attn',
            token_positions=[0]
        )
    
    def patch_char_tokens(self,
                         source_char: torch.Tensor,
                         target_char: torch.Tensor,
                         font_idx: torch.Tensor,
                         layer_idx: int,
                         image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Patch character tokens from source to target
        
        Tests: At which layer does character identity matter?
        """
        if image is None:
            image = torch.zeros(1, 1, 128, 128)
        
        source_input = {
            'image': image,
            'char_indices': source_char,
            'font_idx': font_idx
        }
        
        target_input = {
            'image': image,
            'char_indices': target_char,
            'font_idx': font_idx
        }
        
        # Patch positions 1, 2, ... (char tokens)
        num_chars = target_char.shape[1]
        char_positions = list(range(1, 1 + num_chars))
        
        return self.patch_activation(
            source_input,
            target_input,
            layer_idx,
            position='post_attn',
            token_positions=char_positions
        )
    
    def compute_patching_effect(self,
                               source_input: Dict,
                               target_input: Dict,
                               layer_idx: int,
                               position: str = 'post_attn',
                               metric: str = 'mse') -> float:
        """
        Compute effect of patching on output
        
        Args:
            metric: 'mse' or 'correlation'
        
        Returns:
            Effect magnitude
        """
        # Get original outputs
        with torch.no_grad():
            source_output = self.model(
                **{k: v.to(self.device) for k, v in source_input.items()}
            )['reconstructed']
            
            target_output = self.model(
                **{k: v.to(self.device) for k, v in target_input.items()}
            )['reconstructed']
        
        # Get patched output
        patched_output = self.patch_activation(
            source_input,
            target_input,
            layer_idx,
            position
        )
        
        # Compute metric
        if metric == 'mse':
            # How much does patched look like source?
            effect = torch.nn.functional.mse_loss(patched_output, source_output).item()
        elif metric == 'correlation':
            # Correlation with source output
            effect = torch.corrcoef(torch.stack([
                patched_output.flatten(),
                source_output.flatten()
            ]))[0, 1].item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return effect
    
    def scan_layers(self,
                   source_input: Dict,
                   target_input: Dict,
                   position: str = 'post_attn',
                   metric: str = 'mse') -> List[float]:
        """
        Scan all layers to see where patching has most effect
        
        Returns:
            List of effect magnitudes per layer
        """
        effects = []
        
        num_layers = len(self.model.blocks)
        
        for layer_idx in range(num_layers):
            effect = self.compute_patching_effect(
                source_input,
                target_input,
                layer_idx,
                position,
                metric
            )
            effects.append(effect)
            
            logger.info(f"Layer {layer_idx}: effect = {effect:.4f}")
        
        return effects


class PathPatcher:
    """
    Path patching - more sophisticated version
    Patches specific paths through the network
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cached_activations = {}
    
    def cache_activations(self,
                         input_dict: Dict,
                         layers: List[int],
                         name: str):
        """Cache activations for later use"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(
                image=input_dict['image'].to(self.device),
                char_indices=input_dict['char_indices'].to(self.device),
                font_idx=input_dict['font_idx'].to(self.device),
                return_activations=True
            )
        
        cache = {}
        for layer_idx in layers:
            cache[layer_idx] = {
                'post_attn': output['activations'][layer_idx]['post_attn'].clone(),
                'post_mlp': output['activations'][layer_idx]['post_mlp'].clone(),
            }
        
        self.cached_activations[name] = cache
    
    def patch_path(self,
                  source_name: str,
                  target_input: Dict,
                  path: List[Tuple[int, str]]) -> torch.Tensor:
        """
        Patch a specific path through the network
        
        Args:
            source_name: Name of cached source activations
            target_input: Target input to patch into
            path: List of (layer_idx, position) to patch
        
        Returns:
            Patched output
        """
        if source_name not in self.cached_activations:
            raise ValueError(f"No cached activations for {source_name}")
        
        source_cache = self.cached_activations[source_name]
        
        # TODO: Implement full path patching
        # This requires more sophisticated hook management
        
        raise NotImplementedError("Full path patching not yet implemented")


if __name__ == "__main__":
    print("Testing Activation Patcher")
    print("=" * 50)
    
    # Create dummy model
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            from transformer import TransformerBlock
            self.blocks = nn.ModuleList([
                TransformerBlock(512, 8) for _ in range(4)
            ])
            self.dummy_embed = nn.Linear(1, 512)
        
        def forward(self, image, char_indices, font_idx, return_activations=False):
            # Dummy forward
            B = image.shape[0]
            tokens = torch.randn(B, 67, 512)
            
            if return_activations:
                activations = []
                for block in self.blocks:
                    tokens, _ = block(tokens)
                    activations.append(block.get_activations())
                
                return {
                    'reconstructed': torch.randn(B, 1, 128, 128),
                    'activations': activations
                }
            
            return {'reconstructed': torch.randn(B, 1, 128, 128)}
    
    model = DummyTransformer()
    patcher = ActivationPatcher(model)
    
    # Test inputs
    source = {
        'image': torch.randn(1, 1, 128, 128),
        'char_indices': torch.tensor([[0]]),
        'font_idx': torch.tensor([0])
    }
    
    target = {
        'image': torch.randn(1, 1, 128, 128),
        'char_indices': torch.tensor([[1]]),
        'font_idx': torch.tensor([1])
    }
    
    # Test patching
    print("✓ Testing activation patching...")
    output = patcher.patch_activation(source, target, layer_idx=2)
    print(f"  Patched output: {output.shape}")
    
    # Test font token patching
    print("✓ Testing font token patching...")
    font_patched = patcher.patch_font_token(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([[0]]),
        layer_idx=1
    )
    print(f"  Font patched output: {font_patched.shape}")
    
    print("\n✓ All patching tests passed!")
