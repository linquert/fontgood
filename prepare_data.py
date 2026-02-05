"""
Memory-Optimized Data Preparation for Pure Typographic Transformer
OPTIMIZED for low-RAM environments like Google Colab

Key improvements:
- Chunked rendering (minimal peak RAM)
- uint8 storage (4x memory reduction)
- Progress monitoring
- RAM usage estimation
- Automatic cleanup
"""

import argparse
import yaml
from pathlib import Path
import sys
import psutil
import os
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import optimized versions first, fall back to original
try:
    from data.renderer import MemoryOptimizedRenderer
    from data.dataset import MemoryMonitor
    USE_OPTIMIZED = True
    print("✓ Using memory-optimized renderer")
except ImportError:
    from data.renderer import SequenceRenderer
    USE_OPTIMIZED = True
    print("⚠ Using standard renderer (memory-optimized not found)")

from data.font_metadata import FontMetadataLoader


def get_ram_usage_mb():
    """Get current RAM usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def estimate_memory_requirements(num_fonts, sequences_per_font, image_size=128):
    """Estimate memory needed for dataset"""
    if USE_OPTIMIZED:
        estimates = MemoryMonitor.estimate_dataset_size(
            num_fonts=num_fonts,
            sequences_per_font=sequences_per_font,
            image_size=image_size,
            use_uint8=True
        )
        return estimates
    else:
        # Manual estimation for standard renderer
        bytes_per_image = image_size * image_size * 4  # float32
        total_images = num_fonts * sequences_per_font
        total_mb = (total_images * bytes_per_image) / (1024 * 1024) * 1.1
        return {
            'total_images': total_images,
            'total_with_overhead_mb': total_mb,
            'storage_type': 'float32',
            'per_font_mb': total_mb / num_fonts
        }


def main():
    parser = argparse.ArgumentParser(
        description='Prepare typographic dataset with memory optimization'
    )
    parser.add_argument('--config', default='../configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_fonts', type=int, default=None,
                       help='Number of fonts (overrides config)')
    parser.add_argument('--skip_render', action='store_true',
                       help='Skip rendering if cache exists')
    parser.add_argument('--chunk_size', type=int, default=10,
                       help='Number of fonts to process at once (memory optimization)')
    parser.add_argument('--storage_format', choices=['pickle', 'hdf5', 'zarr'],
                       default='pickle',
                       help='Storage format (hdf5/zarr for huge datasets)')
    parser.add_argument('--force_cleanup', action='store_true',
                       help='Force garbage collection between chunks')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    
    if not config_path.exists():
        # Try alternative paths
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override num_fonts if specified
    if args.num_fonts is not None:
        config['data']['num_fonts'] = args.num_fonts
    
    num_fonts = config['data']['num_fonts']
    
    print("=" * 80)
    print(f"PREPARING TYPOGRAPHIC DATASET ({num_fonts} fonts)")
    print("=" * 80)
    print(f"Memory optimization: {'ENABLED' if USE_OPTIMIZED else 'DISABLED'}")
    print(f"Chunk size: {args.chunk_size} fonts")
    print(f"Storage format: {args.storage_format}")
    print(f"Initial RAM: {get_ram_usage_mb():.1f} MB")
    print("=" * 80)
    
    # Load fonts
    print("\n[1/4] Loading fonts from Google Fonts...")
    initial_ram = get_ram_usage_mb()
    
    loader = FontMetadataLoader(config['data']['cache_dir'])
    fonts = loader.load_all_fonts(
        min_fonts=config['data']['min_fonts'],
        max_fonts=config['data']['num_fonts']
    )
    
    if not fonts:
        print("ERROR: Failed to load fonts")
        return
    
    fonts_ram = get_ram_usage_mb()
    print(f"✓ Loaded {len(fonts)} fonts")
    print(f"  RAM: {fonts_ram:.1f} MB (Δ {fonts_ram - initial_ram:.1f} MB)")
    
    # Create splits
    print("\n[2/4] Creating splits...")
    splits = loader.create_splits(
        fonts,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        continual_ratio=config['data']['continual_ratio']
    )
    
    for split_name, split_fonts in splits.items():
        print(f"  {split_name}: {len(split_fonts)} fonts")
    
    # Generate sequences
    print("\n[3/4] Generating sequence list...")
    
    # Determine max_sequence_length
    max_seq_len = config['data'].get('max_sequence_length', 3)
    
    # Get common bigrams/trigrams
    common_bigrams = config['data'].get('common_bigrams', None)
    common_trigrams = config['data'].get('common_trigrams', None)
    
    # Estimate sequences
    num_sequences = len(config['data']['characters'])  # Single chars
    if max_seq_len >= 2:
        num_sequences += len(common_bigrams) if common_bigrams else 30
    if max_seq_len >= 3:
        num_sequences += len(common_trigrams) if common_trigrams else 6
    
    print(f"  Estimated sequences per font: ~{num_sequences}")
    
    # Memory estimation
    print("\n  Memory estimation:")
    estimates = estimate_memory_requirements(
        num_fonts=len(fonts),
        sequences_per_font=num_sequences,
        image_size=config['data']['image_size']
    )
    
    print(f"    Total images: {estimates['total_images']:,}")
    print(f"    Storage type: {estimates['storage_type']}")
    print(f"    Dataset size: {estimates['total_with_overhead_mb']:.1f} MB")
    print(f"    Per font: {estimates['per_font_mb']:.2f} MB")
    
    if USE_OPTIMIZED:
        peak_estimate = args.chunk_size * estimates['per_font_mb']
        print(f"    Peak RAM (chunked): ~{peak_estimate:.1f} MB")
    
    # Ask for confirmation if dataset is large
    if estimates['total_with_overhead_mb'] > 1000 and not args.skip_render:
        print(f"\n⚠  Large dataset detected ({estimates['total_with_overhead_mb']:.0f} MB)")
        if not USE_OPTIMIZED:
            print("   Consider using optimized renderer to reduce memory usage")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Create renderer
    if USE_OPTIMIZED:
        renderer = MemoryOptimizedRenderer(
            image_size=config['data']['image_size'],
            font_size=config['data']['font_size'],
            output_dir=config['data']['rendered_dir'],
            storage_format=args.storage_format,
            chunk_size=args.chunk_size
        )
        
        # Generate sequences
        sequences = renderer.generate_sequences(
            characters=config['data']['characters'],
            max_length=max_seq_len,
            common_bigrams=common_bigrams,
            common_trigrams=common_trigrams
        )
    else:
        renderer = SequenceRenderer(
            image_size=config['data']['image_size'],
            font_size=config['data']['font_size'],
            output_dir=config['data']['rendered_dir']
        )
        
        # Generate sequences
        sequences = renderer.generate_sequences(
            characters=config['data']['characters'],
            max_length=max_seq_len,
            common_bigrams=common_bigrams,
            common_trigrams=common_trigrams
        )
    
    print(f"\n✓ Generated {len(sequences)} sequences")
    print(f"  Single chars: {sum(1 for s in sequences if len(s)==1)}")
    print(f"  Bigrams: {sum(1 for s in sequences if len(s)==2)}")
    print(f"  Trigrams: {sum(1 for s in sequences if len(s)==3)}")
    
    # Render
    print("\n[4/4] Rendering images...")
    print(f"  Using {'CHUNKED' if USE_OPTIMIZED else 'STANDARD'} rendering")
    
    for split_name, split_fonts in splits.items():
        cache_file = Path(config['data']['rendered_dir']) / f'{split_name}_dataset.pkl'
        
        if args.skip_render and cache_file.exists():
            print(f"\n✓ Skipping {split_name} (cached)")
            continue
        
        print(f"\n  Rendering {split_name} split ({len(split_fonts)} fonts)...")
        pre_render_ram = get_ram_usage_mb()
        print(f"    Pre-render RAM: {pre_render_ram:.1f} MB")
        
        if USE_OPTIMIZED:
            # Use chunked rendering
            renderer.render_dataset_chunked(split_fonts, sequences, split_name)
            
            # Force cleanup if requested
            if args.force_cleanup:
                gc.collect()
            
            peak_ram = get_ram_usage_mb()
            print(f"    Peak RAM: {peak_ram:.1f} MB (Δ {peak_ram - pre_render_ram:.1f} MB)")
            
            # Load to get sample count
            if args.storage_format == 'pickle':
                import pickle
                with open(cache_file, 'rb') as f:
                    rendered = pickle.load(f)
                total_samples = sum(len(f['sequences']) for f in rendered)
                del rendered
            else:
                total_samples = len(split_fonts) * len(sequences)
            
        else:
            # Standard rendering
            rendered = renderer.render_dataset(split_fonts, sequences, split_name)
            total_samples = sum(len(f['sequences']) for f in rendered)
            
            peak_ram = get_ram_usage_mb()
            print(f"    Peak RAM: {peak_ram:.1f} MB (Δ {peak_ram - pre_render_ram:.1f} MB)")
        
        print(f"  ✓ {split_name}: {len(split_fonts)} fonts, {total_samples} samples")
        
        # Cleanup
        if args.force_cleanup:
            gc.collect()
    
    # Final summary
    final_ram = get_ram_usage_mb()
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nDataset summary:")
    print(f"  Fonts: {len(fonts)}")
    print(f"  Sequences per font: {len(sequences)}")
    print(f"  Total samples: ~{len(fonts) * len(sequences):,}")
    print(f"  Storage format: {args.storage_format}")
    
    print(f"\nMemory usage:")
    print(f"  Initial: {initial_ram:.1f} MB")
    print(f"  Final: {final_ram:.1f} MB")
    print(f"  Delta: {final_ram - initial_ram:.1f} MB")
    
    if USE_OPTIMIZED:
        print(f"\nOptimization stats:")
        print(f"  Chunk size: {args.chunk_size} fonts")
        print(f"  Storage: {estimates['storage_type']}")
        standard_estimate = estimates['total_with_overhead_mb']
        if estimates['storage_type'] == 'uint8':
            standard_estimate *= 4  # Would be 4x if using float32
        savings_pct = (1 - estimates['total_with_overhead_mb'] / standard_estimate) * 100
        print(f"  Memory saved: ~{savings_pct:.0f}%")
    
    print(f"\nOutput directory: {config['data']['rendered_dir']}")
    print(f"\nNext steps:")
    print(f"  1. Verify rendered data:")
    print(f"     ls -lh {config['data']['rendered_dir']}")
    print(f"  2. Start training:")
    print(f"     python training/train.py --config {args.config}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)