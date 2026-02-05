"""
Data Preparation for Pure Typographic Transformer
Prepares 500-2000 fonts with multi-character sequences
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.font_metadata import FontMetadataLoader
from data.renderer import SequenceRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.yaml')
    parser.add_argument('--num_fonts', type=int, default=500)
    parser.add_argument('--skip_render', action='store_true')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['data']['num_fonts'] = args.num_fonts
    
    print("=" * 80)
    print(f"PREPARING TYPOGRAPHIC DATASET ({args.num_fonts} fonts)")
    print("=" * 80)
    
    # Load fonts
    print("\n[1/4] Loading fonts from Google Fonts...")
    loader = FontMetadataLoader(config['data']['cache_dir'])
    fonts = loader.load_all_fonts(
        min_fonts=config['data']['min_fonts'],
        max_fonts=config['data']['num_fonts']
    )
    
    if not fonts:
        print("ERROR: Failed to load fonts")
        return
    
    print(f"✓ Loaded {len(fonts)} fonts")
    
    # Create splits
    print("\n[2/4] Creating splits...")
    splits = loader.create_splits(
        fonts,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        continual_ratio=config['data']['continual_ratio']
    )
    
    # Generate sequences
    print("\n[3/4] Generating sequence list...")
    renderer = SequenceRenderer(
        image_size=config['data']['image_size'],
        font_size=config['data']['font_size'],
        output_dir=config['data']['rendered_dir']
    )
    
    sequences = renderer.generate_sequences(
        characters=config['data']['characters'],
        max_length=config['data']['max_sequence_length'],
        common_bigrams=config['data'].get('common_bigrams'),
        common_trigrams=config['data'].get('common_trigrams')
    )
    
    print(f"✓ Generated {len(sequences)} sequences")
    print(f"  Single: {sum(1 for s in sequences if len(s)==1)}")
    print(f"  Bigrams: {sum(1 for s in sequences if len(s)==2)}")
    print(f"  Trigrams: {sum(1 for s in sequences if len(s)==3)}")
    
    # Render
    print("\n[4/4] Rendering images...")
    for split_name, split_fonts in splits.items():
        if args.skip_render and (Path(config['data']['rendered_dir']) / f'{split_name}_dataset.pkl').exists():
            print(f"✓ Skipping {split_name} (cached)")
            continue
        
        rendered = renderer.render_dataset(split_fonts, sequences, split_name)
        total_samples = sum(len(f['sequences']) for f in rendered)
        print(f"✓ {split_name}: {len(rendered)} fonts, {total_samples} samples")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nDataset summary:")
    print(f"  Fonts: {len(fonts)}")
    print(f"  Sequences per font: {len(sequences)}")
    print(f"  Total samples: ~{len(fonts) * len(sequences)}")
    print(f"\nNext: python training/train.py")


if __name__ == "__main__":
    main()
