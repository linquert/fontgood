"""
Linear Probes for Mechanistic Interpretability
Train classifiers on layer activations to detect learned features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Simple linear probe for classification"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProbeTrainer:
    """
    Train and evaluate linear probes on model activations
    
    Probe targets:
    - Character identity (52-way)
    - Uppercase vs lowercase (binary)
    - Vowel vs consonant (binary)
    - Serif vs sans (binary)
    - Weight category (3-way: light, regular, bold)
    - Slant (binary: normal, italic)
    - Width category (3-way: condensed, normal, wide)
    - Sequence length (3-way: 1, 2, 3 chars)
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probes = {}
        self.results = {}
    
    def extract_activations(self,
                          model: nn.Module,
                          dataloader: DataLoader,
                          layer_idx: int,
                          position: str = 'post_attn',
                          max_samples: int = 5000) -> Tuple[torch.Tensor, Dict]:
        """
        Extract activations from a specific layer
        
        Args:
            model: The model
            dataloader: Data loader
            layer_idx: Which transformer layer (0-7)
            position: 'post_attn', 'post_mlp', 'input'
            max_samples: Maximum samples to collect
        
        Returns:
            activations: (N, D) tensor
            labels: Dict of label tensors
        """
        model.eval()
        
        all_activations = []
        all_labels = {
            'char_id': [],
            'uppercase': [],
            'vowel': [],
            'serif': [],
            'weight': [],
            'slant': [],
            'width': [],
            'seq_len': []
        }
        
        samples_collected = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= max_samples:
                    break
                
                images = batch['image'].to(self.device)
                char_indices = batch['char_indices'].to(self.device)
                font_idx = batch['font_idx'].to(self.device)
                font_attrs = batch['font_attrs'].to(self.device)
                
                # Forward pass with activations
                output = model(
                    image=images,
                    char_indices=char_indices,
                    font_idx=font_idx,
                    return_activations=True
                )
                
                # Get activations from specified layer
                layer_acts = output['activations'][layer_idx][position]
                
                # Use first character token activations
                # Position 1 (after font token)
                act = layer_acts[:, 1, :]  # (B, D)
                
                all_activations.append(act.cpu())
                
                # Collect labels
                # Character identity (first char only)
                char_ids = char_indices[:, 0]
                all_labels['char_id'].append(char_ids.cpu())
                
                # Uppercase vs lowercase
                uppercase = (char_ids < 26).long()
                all_labels['uppercase'].append(uppercase.cpu())
                
                # Vowel vs consonant
                vowels = torch.tensor([0, 4, 8, 14, 20, 26, 30, 34, 40, 46])  # A, E, I, O, U, a, e, i, o, u
                is_vowel = torch.isin(char_ids, vowels.to(char_ids.device)).long()
                all_labels['vowel'].append(is_vowel.cpu())
                
                # Font attributes
                # Serif vs sans
                serif = (font_attrs[:, 0] > 0.5).long()
                all_labels['serif'].append(serif.cpu())
                
                # Weight (light < 0.4, regular 0.4-0.7, bold > 0.7)
                weight = font_attrs[:, 1]
                weight_cat = torch.zeros_like(weight, dtype=torch.long)
                weight_cat[weight < 0.4] = 0
                weight_cat[(weight >= 0.4) & (weight < 0.7)] = 1
                weight_cat[weight >= 0.7] = 2
                all_labels['weight'].append(weight_cat.cpu())
                
                # Slant
                slant = (font_attrs[:, 3] > 0.5).long()
                all_labels['slant'].append(slant.cpu())
                
                # Width
                width = font_attrs[:, 2]
                width_cat = torch.zeros_like(width, dtype=torch.long)
                width_cat[width < 0.4] = 0
                width_cat[(width >= 0.4) & (width < 0.7)] = 1
                width_cat[width >= 0.7] = 2
                all_labels['width'].append(width_cat.cpu())
                
                # Sequence length
                seq_len = batch['seq_lengths'] - 1  # 0-indexed (0, 1, 2 for lengths 1, 2, 3)
                all_labels['seq_len'].append(seq_len.cpu())
                
                samples_collected += images.shape[0]
        
        # Concatenate
        activations = torch.cat(all_activations, dim=0)
        for key in all_labels:
            all_labels[key] = torch.cat(all_labels[key], dim=0)
        
        logger.info(f"Extracted {activations.shape[0]} activations from layer {layer_idx}, position {position}")
        
        return activations, all_labels
    
    def train_probe(self,
                   activations: torch.Tensor,
                   labels: torch.Tensor,
                   num_classes: int,
                   probe_name: str,
                   epochs: int = 50,
                   lr: float = 0.001) -> Dict[str, float]:
        """
        Train a linear probe
        
        Returns:
            Dict with accuracy and other metrics
        """
        # Split train/val
        n = activations.shape[0]
        n_train = int(0.8 * n)
        
        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train = activations[train_idx]
        y_train = labels[train_idx]
        X_val = activations[val_idx]
        y_val = labels[val_idx]
        
        # Create probe
        probe = LinearProbe(activations.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        
        # Train
        best_acc = 0.0
        best_probe_state = None
        
        for epoch in range(epochs):
            probe.train()
            
            # Simple batch training
            batch_size = 256
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                
                logits = probe(batch_X)
                loss = F.cross_entropy(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validate
            if epoch % 10 == 0:
                probe.eval()
                with torch.no_grad():
                    val_logits = probe(X_val.to(self.device))
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == y_val.to(self.device)).float().mean().item()
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_probe_state = probe.state_dict().copy()
        
        # Load best probe
        probe.load_state_dict(best_probe_state)
        
        # Final evaluation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val.to(self.device))
            val_preds = val_logits.argmax(dim=1).cpu()
            
            accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
            
            if num_classes == 2:
                f1 = f1_score(y_val.numpy(), val_preds.numpy())
            else:
                f1 = f1_score(y_val.numpy(), val_preds.numpy(), average='macro')
        
        # Store probe
        self.probes[probe_name] = probe
        
        logger.info(f"Probe {probe_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'num_classes': num_classes
        }
    
    def train_all_probes(self,
                        model: nn.Module,
                        dataloader: DataLoader,
                        layers: List[int] = [0, 2, 4, 6],
                        position: str = 'post_attn') -> Dict:
        """
        Train probes on multiple layers
        
        Returns:
            Dict mapping (layer, probe_name) -> metrics
        """
        all_results = {}
        
        for layer_idx in layers:
            logger.info(f"\nProcessing layer {layer_idx}...")
            
            # Extract activations
            activations, labels = self.extract_activations(
                model, dataloader, layer_idx, position
            )
            
            # Train each probe
            probe_configs = [
                ('char_id', labels['char_id'], 52),
                ('uppercase', labels['uppercase'], 2),
                ('vowel', labels['vowel'], 2),
                ('serif', labels['serif'], 2),
                ('weight', labels['weight'], 3),
                ('slant', labels['slant'], 2),
                ('width', labels['width'], 3),
                ('seq_len', labels['seq_len'], 3),
            ]
            
            for probe_name, probe_labels, num_classes in probe_configs:
                # Skip if not enough samples per class
                unique_labels = torch.unique(probe_labels)
                if len(unique_labels) < num_classes:
                    logger.warning(f"Skipping {probe_name} - not enough classes")
                    continue
                
                full_name = f"layer{layer_idx}_{probe_name}"
                
                results = self.train_probe(
                    activations,
                    probe_labels,
                    num_classes,
                    full_name
                )
                
                all_results[(layer_idx, probe_name)] = results
        
        self.results = all_results
        return all_results
    
    def get_accuracy_by_layer(self, probe_name: str) -> Dict[int, float]:
        """Get accuracy across layers for a specific probe"""
        accuracies = {}
        
        for (layer, name), metrics in self.results.items():
            if name == probe_name:
                accuracies[layer] = metrics['accuracy']
        
        return accuracies
    
    def save_results(self, path: str):
        """Save probe results"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'probes': {k: v.state_dict() for k, v in self.probes.items()}
            }, f)
        logger.info(f"Saved probe results to {path}")


if __name__ == "__main__":
    # Test probe trainer
    print("Testing Linear Probes")
    print("=" * 50)
    
    # Create dummy data
    n_samples = 1000
    dim = 512
    
    activations = torch.randn(n_samples, dim)
    labels = torch.randint(0, 52, (n_samples,))
    
    trainer = ProbeTrainer()
    
    results = trainer.train_probe(
        activations,
        labels,
        num_classes=52,
        probe_name='test_char_id',
        epochs=30
    )
    
    print(f"\n✓ Probe trained:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    
    print("\n✓ Probe tests passed!")
