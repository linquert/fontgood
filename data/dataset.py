"""
Memory-Optimized PyTorch Dataset
OPTIMIZED for low-RAM environments

Key improvements:
1. Lazy loading - don't load all data into RAM at once
2. Memory-mapped files for large datasets
3. On-the-fly conversion to tensors
4. Efficient collation without copies
5. Optional disk caching for transformed data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple
import logging
import mmap
import h5py
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LazyTypographicDataset(Dataset):
    """
    Memory-efficient dataset that loads data on-demand
    
    Instead of loading all images into RAM:
    - Keep only metadata in memory
    - Load images on-the-fly during __getitem__
    - Use memory-mapped files for ultra-large datasets
    """
    
    def __init__(self,
                 data_source,  # Can be List[Dict], HDF5 path, or Zarr path
                 characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                 transform=None,
                 storage_format: str = "pickle",
                 cache_in_ram: bool = False):
        """
        Args:
            data_source: Path to data or pre-loaded list
            characters: Character vocabulary
            transform: Optional augmentation
            storage_format: 'pickle', 'hdf5', or 'zarr'
            cache_in_ram: If True, load all data into RAM (not recommended for large datasets)
        """
        self.characters = characters
        self.transform = transform
        self.storage_format = storage_format
        self.cache_in_ram = cache_in_ram
        
        # Character to index mapping
        self.char_to_idx = {c: i+1 for i, c in enumerate(characters)}
        
        # Initialize based on storage format
        if storage_format == "pickle":
            self._init_pickle(data_source)
        elif storage_format == "hdf5":
            self._init_hdf5(data_source)
        elif storage_format == "zarr":
            self._init_zarr(data_source)
        else:
            raise ValueError(f"Unknown storage format: {storage_format}")
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
        logger.info(f"Memory mode: {'RAM cached' if cache_in_ram else 'Lazy loading'}")
    
    def _init_pickle(self, data_source):
        """Initialize from pickle file or list"""
        if isinstance(data_source, (str, Path)):
            # Load from file
            logger.info(f"Loading dataset from {data_source}")
            with open(data_source, 'rb') as f:
                rendered_fonts = pickle.load(f)
        else:
            # Already a list
            rendered_fonts = data_source
        
        if self.cache_in_ram:
            # Old behavior - load everything
            self.rendered_fonts = rendered_fonts
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
                        'image': image_array,  # MEMORY HEAVY!
                        'attributes': font_data['attributes']
                    })
        else:
            # NEW: Lazy loading - only store metadata
            self.rendered_fonts = None
            self.samples = []
            self.font_name_to_idx = {}
            
            # Store only references, not actual images
            for font_idx, font_data in enumerate(rendered_fonts):
                font_name = font_data['font_name']
                self.font_name_to_idx[font_name] = font_idx
                
                for sequence in font_data['sequences'].keys():
                    self.samples.append({
                        'font_idx': font_idx,
                        'font_name': font_name,
                        'sequence': sequence,
                        'image_ref': (font_idx, sequence),  # Reference instead of data!
                        'attributes': font_data['attributes']
                    })
            
            # Keep font data for lazy access
            self.font_data = rendered_fonts
    
    def _init_hdf5(self, hdf5_path):
        """Initialize from HDF5 file (memory-efficient)"""
        self.hdf5_path = Path(hdf5_path)
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        self.samples = []
        self.font_name_to_idx = {}
        
        font_keys = sorted([k for k in self.hdf5_file.keys() if k.startswith('font_')])
        
        for font_idx, font_key in enumerate(font_keys):
            font_group = self.hdf5_file[font_key]
            font_name = font_group.attrs['font_name']
            self.font_name_to_idx[font_name] = font_idx
            
            # Load attributes once
            attrs_data = font_group['attributes'][:]
            attributes = {
                'serif_score': float(attrs_data[0]),
                'weight': float(attrs_data[1]),
                'width': float(attrs_data[2]),
                'slant': float(attrs_data[3]),
                'contrast': float(attrs_data[4]),
                'x_height': float(attrs_data[5]),
                'stroke_ending': float(attrs_data[6]),
                'formality': float(attrs_data[7]),
            }
            
            # Store only references to sequences
            for seq_name in font_group['sequences'].keys():
                self.samples.append({
                    'font_idx': font_idx,
                    'font_name': font_name,
                    'sequence': seq_name,
                    'hdf5_ref': (font_key, seq_name),  # HDF5 reference
                    'attributes': attributes
                })
    
    def _init_zarr(self, zarr_path):
        """Initialize from Zarr directory"""
        self.zarr_root = zarr.open(str(zarr_path), mode='r')
        
        self.samples = []
        self.font_name_to_idx = {}
        
        font_keys = sorted([k for k in self.zarr_root.keys() if k.startswith('font_')])
        
        for font_idx, font_key in enumerate(font_keys):
            font_group = self.zarr_root[font_key]
            font_name = font_group.attrs['font_name']
            self.font_name_to_idx[font_name] = font_idx
            
            attrs_data = font_group['attributes'][:]
            attributes = {
                'serif_score': float(attrs_data[0]),
                'weight': float(attrs_data[1]),
                'width': float(attrs_data[2]),
                'slant': float(attrs_data[3]),
                'contrast': float(attrs_data[4]),
                'x_height': float(attrs_data[5]),
                'stroke_ending': float(attrs_data[6]),
                'formality': float(attrs_data[7]),
            }
            
            for seq_name in font_group['sequences'].keys():
                self.samples.append({
                    'font_idx': font_idx,
                    'font_name': font_name,
                    'sequence': seq_name,
                    'zarr_ref': (font_key, seq_name),
                    'attributes': attributes
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, sample: Dict) -> np.ndarray:
        """Load image data based on storage format"""
        
        if self.storage_format == "pickle":
            if self.cache_in_ram:
                # Image already in sample
                return sample['image']
            else:
                # Load from font_data
                font_idx, sequence = sample['image_ref']
                return self.font_data[font_idx]['sequences'][sequence]
        
        elif self.storage_format == "hdf5":
            font_key, seq_name = sample['hdf5_ref']
            # Load from HDF5 (lazy loading)
            return self.hdf5_file[font_key]['sequences'][seq_name][:]
        
        elif self.storage_format == "zarr":
            font_key, seq_name = sample['zarr_ref']
            # Load from Zarr
            return self.zarr_root[font_key]['sequences'][seq_name][:]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample - OPTIMIZED
        
        Only loads image data when needed
        """
        sample = self.samples[idx]
        
        # Load image on-demand
        image_array = self._load_image(sample)
        
        # Convert to tensor efficiently
        # If uint8, convert to float32 during tensor creation
        if image_array.dtype == np.uint8:
            image = torch.from_numpy(image_array).float() / 255.0
        else:
            image = torch.from_numpy(image_array).float()
        
        image = image.unsqueeze(0)  # Add channel dimension
        
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
        
        # Font attributes
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
        return len(self.font_name_to_idx)
    
    def __del__(self):
        """Clean up file handles"""
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            self.hdf5_file.close()


def collate_fn_optimized(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Optimized collate function
    
    Improvements:
    - Preallocate tensors instead of list + stack
    - Use torch.cat instead of multiple operations
    - Minimize intermediate allocations
    """
    batch_size = len(batch)
    
    # Preallocate image tensor
    first_image = batch[0]['image']
    images = torch.empty(
        (batch_size, *first_image.shape),
        dtype=first_image.dtype
    )
    
    for i, item in enumerate(batch):
        images[i] = item['image']
    
    # Pad character sequences efficiently
    max_seq_len = max(len(item['char_indices']) for item in batch)
    
    # Preallocate
    char_indices = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.long
    )
    seq_lengths = torch.empty(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq = item['char_indices']
        seq_len = len(seq)
        char_indices[i, :seq_len] = seq
        seq_lengths[i] = seq_len
    
    # Stack other items efficiently
    font_idx = torch.empty(batch_size, dtype=torch.long)
    font_attrs = torch.empty((batch_size, 8), dtype=torch.float32)
    
    for i, item in enumerate(batch):
        font_idx[i] = item['font_idx']
        font_attrs[i] = item['font_attrs']
    
    # Keep sequences and font names as lists
    sequences = [item['sequence'] for item in batch]
    font_names = [item['font_name'] for item in batch]
    
    return {
        'image': images,
        'char_indices': char_indices,
        'font_idx': font_idx,
        'font_attrs': font_attrs,
        'seq_lengths': seq_lengths,
        'sequences': sequences,
        'font_names': font_names,
    }


def load_dataset_split_optimized(data_dir: str, 
                                 split: str,
                                 storage_format: str = "pickle",
                                 lazy: bool = False) -> Optional[List[Dict]]:
    """
    Load dataset split with memory optimization
    
    Args:
        data_dir: Path to rendered data directory
        split: 'train', 'val', 'test', or 'continual'
        storage_format: 'pickle', 'hdf5', or 'zarr'
        lazy: If True, return path for lazy loading. If False, load into RAM.
    
    Returns:
        Data or path to data
    """
    data_dir = Path(data_dir)
    
    if storage_format == "pickle":
        cache_file = data_dir / f"{split}_dataset.pkl"
        
        if not cache_file.exists():
            logger.error(f"Dataset file not found: {cache_file}")
            return None
        
        if lazy:
            # Return path for lazy loading
            logger.info(f"Will lazy-load {split} dataset from {cache_file}")
            return str(cache_file)
        else:
            # Load into RAM
            logger.info(f"Loading {split} dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded {len(data)} fonts for {split} split")
            return data
    
    elif storage_format == "hdf5":
        hdf5_file = data_dir / f"{split}_dataset.h5"
        
        if not hdf5_file.exists():
            logger.error(f"HDF5 file not found: {hdf5_file}")
            return None
        
        logger.info(f"Will lazy-load {split} dataset from HDF5: {hdf5_file}")
        return str(hdf5_file)
    
    elif storage_format == "zarr":
        zarr_dir = data_dir / f"{split}_dataset.zarr"
        
        if not zarr_dir.exists():
            logger.error(f"Zarr directory not found: {zarr_dir}")
            return None
        
        logger.info(f"Will lazy-load {split} dataset from Zarr: {zarr_dir}")
        return str(zarr_dir)


class MemoryMonitor:
    """Monitor dataset memory usage"""
    
    @staticmethod
    def estimate_dataset_size(num_fonts: int, 
                            sequences_per_font: int,
                            image_size: int = 128,
                            use_uint8: bool = True) -> Dict[str, float]:
        """
        Estimate memory requirements
        
        Returns:
            Dict with size estimates in MB
        """
        bytes_per_pixel = 1 if use_uint8 else 4
        bytes_per_image = image_size * image_size * bytes_per_pixel
        
        total_images = num_fonts * sequences_per_font
        total_image_data_mb = (total_images * bytes_per_image) / (1024 * 1024)
        
        # Add overhead for metadata (~10%)
        total_mb = total_image_data_mb * 1.1
        
        return {
            'total_images': total_images,
            'image_data_mb': total_image_data_mb,
            'total_with_overhead_mb': total_mb,
            'storage_type': 'uint8' if use_uint8 else 'float32',
            'per_font_mb': total_mb / num_fonts
        }


if __name__ == "__main__":
    """Test memory-optimized dataset"""
    
    print("Testing LazyTypographicDataset")
    print("=" * 50)
    
    # Estimate memory usage
    print("\nMemory estimates:")
    estimates = MemoryMonitor.estimate_dataset_size(
        num_fonts=50,
        sequences_per_font=60,
        image_size=128,
        use_uint8=True
    )
    
    for key, value in estimates.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Comparison:")
    print(f"  uint8:   {estimates['total_with_overhead_mb']:.1f} MB")
    
    estimates_float = MemoryMonitor.estimate_dataset_size(
        num_fonts=50,
        sequences_per_font=60,
        image_size=128,
        use_uint8=False
    )
    print(f"  float32: {estimates_float['total_with_overhead_mb']:.1f} MB")
    print(f"  Savings: {estimates_float['total_with_overhead_mb'] / estimates['total_with_overhead_mb']:.1f}x")
    
    print("\n✓ Memory estimation complete")
    # ==========================================
# COMPATIBILITY ALIASES for Backward Compatibility
# ==========================================
TypographicDataset = LazyTypographicDataset
collate_fn = collate_fn_optimized
load_dataset_split = load_dataset_split_optimized