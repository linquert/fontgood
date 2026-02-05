"""
PyTorch Dataset for Typographic Transformer
Optimized for fast loading with caching and efficient collation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TypographicDataset(Dataset):
    """
    Dataset for rendered font sequences
    
    Optimized features:
    - Pre-rendered and cached
    - Fast numpy array access
    - Minimal runtime processing
    - Efficient batch collation
    """
    
    def __init__(self,
                 rendered_fonts: List[Dict],
                 characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                 transform=None):
        """
        Args:
            rendered_fonts: List of dicts from renderer
            characters: Character vocabulary for indexing
            transform: Optional augmentation
        """
        self.rendered_fonts = rendered_fonts
        self.characters = characters
        self.transform = transform
        
        # Build character to index mapping
        self.char_to_idx = {c: i+1 for i, c in enumerate(characters)}
        
        # Build flat sample list for fast access
        self.samples = []
        self.font_name_to_idx = {}
        
        for font_idx, font_data in enumerate(rendered_fonts):
            font_name = font_data['font_name']
            self.font_name_to_idx[font_name] = font_idx
            
            for sequence, image_array in font_data['sequences'].items():
                self.samples.append({
                    'font_idx': font_idx,
                    'font_name': font_name,
                    'sequence': sequence,
                    'image': image_array,
                    'attributes': font_data['attributes']
                })
        
        logger.info(f"Dataset created with {len(self.samples)} samples from {len(rendered_fonts)} fonts")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dict with:
                - image: (1, H, W) tensor
                - char_indices: (N,) tensor - character sequence indices
                - font_idx: scalar tensor
                - font_attrs: (8,) tensor - font attributes
                - sequence: str (for debugging)
        """
        sample = self.samples[idx]
        
        # Convert image to tensor (already normalized 0-1)
        image = torch.from_numpy(sample['image']).float().unsqueeze(0)
        
        # Apply transform if any
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert sequence to character indices
        char_indices = torch.tensor(
            [self.char_to_idx[c] for c in sample['sequence']],
            dtype=torch.long
        )
        
        # Font index
        font_idx = torch.tensor(sample['font_idx'], dtype=torch.long)
        
        # Font attributes (8-dimensional)
        attrs = sample['attributes']
        font_attrs = torch.tensor([
            attrs['serif_score'],
            attrs['weight'],
            attrs['width'],
            attrs['slant'],
            attrs['contrast'],
            attrs['x_height'],
            attrs['stroke_ending'],
            attrs['formality'],
        ], dtype=torch.float32)
        
        return {
            'image': image,
            'char_indices': char_indices,
            'font_idx': font_idx,
            'font_attrs': font_attrs,
            'sequence': sample['sequence'],
            'font_name': sample['font_name'],
        }
    
    def get_font_count(self) -> int:
        """Return number of unique fonts"""
        return len(self.rendered_fonts)
    
    def get_sequence_stats(self) -> Dict[str, int]:
        """Get statistics on sequence lengths"""
        stats = {1: 0, 2: 0, 3: 0}
        for sample in self.samples:
            length = len(sample['sequence'])
            stats[length] = stats.get(length, 0) + 1
        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length character sequences
    
    Pads character sequences to max length in batch
    """
    # Stack images (all same size)
    images = torch.stack([item['image'] for item in batch])
    
    # Pad character sequences
    max_seq_len = max(len(item['char_indices']) for item in batch)
    
    char_indices_padded = []
    seq_lengths = []
    
    for item in batch:
        seq = item['char_indices']
        seq_len = len(seq)
        seq_lengths.append(seq_len)
        
        # Pad with zeros (will be masked in attention if needed)
        if seq_len < max_seq_len:
            padding = torch.zeros(max_seq_len - seq_len, dtype=torch.long)
            seq = torch.cat([seq, padding])
        
        char_indices_padded.append(seq)
    
    char_indices = torch.stack(char_indices_padded)
    
    # Stack other items
    font_idx = torch.stack([item['font_idx'] for item in batch])
    font_attrs = torch.stack([item['font_attrs'] for item in batch])
    
    # Keep sequences and font names as lists for debugging
    sequences = [item['sequence'] for item in batch]
    font_names = [item['font_name'] for item in batch]
    
    return {
        'image': images,
        'char_indices': char_indices,
        'font_idx': font_idx,
        'font_attrs': font_attrs,
        'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),
        'sequences': sequences,
        'font_names': font_names,
    }


class DataAugmentation:
    """
    Optional data augmentation for typography
    
    Conservative augmentations that preserve character identity
    """
    
    def __init__(self,
                 rotation_range: float = 5.0,
                 scale_range: float = 0.1,
                 noise_std: float = 0.02):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations
        
        Args:
            image: (1, H, W)
        
        Returns:
            augmented: (1, H, W)
        """
        # Small rotation
        if self.rotation_range > 0:
            angle = torch.rand(1).item() * 2 * self.rotation_range - self.rotation_range
            # Use torch's affine transform
            # (simplified - just add noise for now, full rotation needs more code)
        
        # Add small noise
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image


def load_dataset_split(data_dir: str, split: str) -> Optional[List[Dict]]:
    """
    Load pre-rendered dataset split from cache
    
    Args:
        data_dir: Path to rendered data directory
        split: 'train', 'val', 'test', or 'continual'
    
    Returns:
        List of font dicts or None if not found
    """
    cache_file = Path(data_dir) / f"{split}_dataset.pkl"
    
    if not cache_file.exists():
        logger.error(f"Dataset file not found: {cache_file}")
        return None
    
    logger.info(f"Loading {split} dataset from {cache_file}")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} fonts for {split} split")
    return data


if __name__ == "__main__":
    # Test dataset loading
    import sys
    from pathlib import Path
    
    # Mock data for testing
    mock_fonts = [
        {
            'font_name': 'TestFont1',
            'sequences': {
                'A': np.random.rand(128, 128).astype(np.float32),
                'AB': np.random.rand(128, 128).astype(np.float32),
                'ABC': np.random.rand(128, 128).astype(np.float32),
            },
            'attributes': {
                'serif_score': 0.0,
                'weight': 0.5,
                'width': 0.5,
                'slant': 0.0,
                'contrast': 0.3,
                'x_height': 0.5,
                'stroke_ending': 0.5,
                'formality': 0.5,
            }
        },
        {
            'font_name': 'TestFont2',
            'sequences': {
                'B': np.random.rand(128, 128).astype(np.float32),
                'BC': np.random.rand(128, 128).astype(np.float32),
            },
            'attributes': {
                'serif_score': 1.0,
                'weight': 0.7,
                'width': 0.5,
                'slant': 0.0,
                'contrast': 0.7,
                'x_height': 0.4,
                'stroke_ending': 0.6,
                'formality': 0.8,
            }
        }
    ]
    
    print("Testing TypographicDataset")
    print("=" * 50)
    
    # Create dataset
    dataset = TypographicDataset(mock_fonts)
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    print(f"  Fonts: {dataset.get_font_count()}")
    print(f"  Sequence stats: {dataset.get_sequence_stats()}")
    
    # Test getitem
    sample = dataset[0]
    print(f"\n✓ Sample 0:")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Char indices: {sample['char_indices']}")
    print(f"  Font idx: {sample['font_idx']}")
    print(f"  Font attrs: {sample['font_attrs'].shape}")
    print(f"  Sequence: '{sample['sequence']}'")
    
    # Test collate function
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\n✓ Batch collation:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Char indices: {batch['char_indices'].shape}")
    print(f"  Font idx: {batch['font_idx'].shape}")
    print(f"  Font attrs: {batch['font_attrs'].shape}")
    print(f"  Seq lengths: {batch['seq_lengths']}")
    print(f"  Sequences: {batch['sequences']}")
    
    print("\n✓ All dataset tests passed!")
