"""
Main Interpretability Analysis Script
Run comprehensive mechanistic interpretability experiments
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent))

from typo_transformer import TypographicTransformer
from dataset import load_dataset_split, TypographicDataset, collate_fn
from torch.utils.data import DataLoader
from probes import ProbeTrainer
from attention_analysis import AttentionAnalyzer
from patching import ActivationPatcher
from visualization import InterpVisualizer
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    model = TypographicTransformer(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model


def run_probe_experiments(model, dataloader, device, save_dir):
    """Run linear probe experiments"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING LINEAR PROBE EXPERIMENTS")
    logger.info("=" * 80)
    
    probe_trainer = ProbeTrainer(device)
    
    # Train probes on all layers
    layers = list(range(len(model.blocks)))
    results = probe_trainer.train_all_probes(
        model,
        dataloader,
        layers=layers,
        position='post_attn'
    )
    
    # Save results
    results_path = Path(save_dir) / 'probe_results.pkl'
    probe_trainer.save_results(results_path)
    
    # Visualize
    viz = InterpVisualizer(save_dir)
    viz.plot_probe_accuracies(results)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("PROBE RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for probe_name in ['char_id', 'serif', 'weight', 'uppercase']:
        accs = probe_trainer.get_accuracy_by_layer(probe_name)
        logger.info(f"\n{probe_name}:")
        for layer, acc in sorted(accs.items()):
            logger.info(f"  Layer {layer}: {acc:.3f}")
    
    return results


def run_attention_experiments(model, dataloader, device, save_dir):
    """Run attention analysis experiments"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ATTENTION ANALYSIS")
    logger.info("=" * 80)
    
    analyzer = AttentionAnalyzer(model, device)
    
    # Get a batch
    batch = next(iter(dataloader))
    images = batch['image'][:8]
    char_indices = batch['char_indices'][:8]
    font_idx = batch['font_idx'][:8]
    
    # Get attention patterns
    attention_patterns = analyzer.get_attention_patterns(images, char_indices, font_idx)
    
    logger.info(f"Collected attention from {len(attention_patterns)} layers")
    
    # Compute layer statistics
    layer_stats = analyzer.compute_layer_statistics(attention_patterns)
    
    # Analyze token importance
    importance = analyzer.analyze_token_importance(attention_patterns, num_special_tokens=3)
    
    # Analyze head specialization
    head_stats = analyzer.analyze_head_specialization(attention_patterns, num_special_tokens=3)
    
    # Visualize
    viz = InterpVisualizer(save_dir)
    viz.plot_attention_entropy(layer_stats)
    viz.plot_token_importance(importance)
    viz.plot_head_specialization(head_stats)
    
    # Visualize specific attention maps
    for layer_idx in [0, 2, 4, 6]:
        analyzer.visualize_attention_map(
            attention_patterns[layer_idx],
            layer_idx=layer_idx,
            head_idx=0,
            sample_idx=0,
            save_path=str(Path(save_dir) / f'attention_layer{layer_idx}_head0.png')
        )
    
    # Visualize attention flow
    analyzer.visualize_attention_flow(
        attention_patterns,
        sample_idx=0,
        head_idx=0,
        save_path=str(Path(save_dir) / 'attention_flow.png')
    )
    
    # Save results
    results = {
        'layer_stats': layer_stats,
        'importance': importance,
        'head_stats': head_stats
    }
    
    results_path = Path(save_dir) / 'attention_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved attention analysis to {results_path}")
    
    return results


def run_patching_experiments(model, dataloader, device, save_dir):
    """Run activation patching experiments"""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING ACTIVATION PATCHING")
    logger.info("=" * 80)
    
    patcher = ActivationPatcher(model, device)
    
    # Get two different samples
    batch = next(iter(dataloader))
    
    source_input = {
        'image': batch['image'][:1],
        'char_indices': batch['char_indices'][:1],
        'font_idx': batch['font_idx'][:1]
    }
    
    target_input = {
        'image': batch['image'][1:2],
        'char_indices': batch['char_indices'][1:2],
        'font_idx': batch['font_idx'][1:2]
    }
    
    logger.info(f"Source: Font {batch['font_names'][0]}, Seq '{batch['sequences'][0]}'")
    logger.info(f"Target: Font {batch['font_names'][1]}, Seq '{batch['sequences'][1]}'")
    
    # Scan all layers
    effects = patcher.scan_layers(
        source_input,
        target_input,
        position='post_attn',
        metric='mse'
    )
    
    logger.info("\nPatching effects by layer:")
    for layer, effect in enumerate(effects):
        logger.info(f"  Layer {layer}: {effect:.4f}")
    
    # Save results
    results_path = Path(save_dir) / 'patching_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({'layer_effects': effects}, f)
    
    return effects


def run_reconstruction_test(model, dataloader, device, save_dir):
    """Test reconstruction quality"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING RECONSTRUCTION")
    logger.info("=" * 80)
    
    model.eval()
    
    batch = next(iter(dataloader))
    images = batch['image'][:8].to(device)
    char_indices = batch['char_indices'][:8].to(device)
    font_idx = batch['font_idx'][:8].to(device)
    
    with torch.no_grad():
        output = model(
            image=images,
            char_indices=char_indices,
            font_idx=font_idx
        )
    
    reconstructed = output['reconstructed'].cpu()
    
    # Compute MSE
    mse = torch.nn.functional.mse_loss(reconstructed, images.cpu()).item()
    logger.info(f"Reconstruction MSE: {mse:.4f}")
    
    # Visualize
    viz = InterpVisualizer(save_dir)
    viz.plot_reconstruction_samples(
        images.cpu(),
        reconstructed,
        batch['sequences'][:8]
    )
    
    return mse


def run_interpolation_test(model, dataloader, device, save_dir):
    """Test font interpolation"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING FONT INTERPOLATION")
    logger.info("=" * 80)
    
    batch = next(iter(dataloader))
    
    # Pick a character
    char_indices = batch['char_indices'][:1]
    
    # Pick two fonts
    font_idx1 = batch['font_idx'][:1]
    font_idx2 = batch['font_idx'][1:2]
    
    font1_name = batch['font_names'][0]
    font2_name = batch['font_names'][1]
    char = batch['sequences'][0]
    
    logger.info(f"Interpolating '{char}' from {font1_name} to {font2_name}")
    
    # Interpolate
    interpolated = model.interpolate_fonts(
        char_indices.to(device),
        font_idx1.to(device),
        font_idx2.to(device),
        num_steps=10
    )
    
    # Visualize
    viz = InterpVisualizer(save_dir)
    viz.plot_font_interpolation(
        interpolated.cpu(),
        font1_name,
        font2_name,
        char
    )


def main():
    parser = argparse.ArgumentParser(description='Run interpretability experiments')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/rendered',
                       help='Path to rendered data')
    parser.add_argument('--save_dir', type=str, default='./interp_results',
                       help='Directory to save results')
    parser.add_argument('--experiments', nargs='+', 
                       default=['probes', 'attention', 'patching', 'reconstruction', 'interpolation'],
                       help='Which experiments to run')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for analysis')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Load data
    logger.info("Loading validation data...")
    val_fonts = load_dataset_split(args.data_dir, 'val')
    
    if val_fonts is None:
        logger.error("Failed to load validation data")
        return
    
    val_dataset = TypographicDataset(val_fonts, config['data']['characters'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    # Run experiments
    results = {}
    
    if 'probes' in args.experiments:
        results['probes'] = run_probe_experiments(model, val_loader, device, save_dir)
    
    if 'attention' in args.experiments:
        results['attention'] = run_attention_experiments(model, val_loader, device, save_dir)
    
    if 'patching' in args.experiments:
        results['patching'] = run_patching_experiments(model, val_loader, device, save_dir)
    
    if 'reconstruction' in args.experiments:
        results['reconstruction'] = run_reconstruction_test(model, val_loader, device, save_dir)
    
    if 'interpolation' in args.experiments:
        run_interpolation_test(model, val_loader, device, save_dir)
    
    # Save all results
    final_results_path = save_dir / 'all_results.pkl'
    with open(final_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL INTERPRETABILITY EXPERIMENTS COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Results saved to {save_dir}")
    logger.info("\nGenerated visualizations:")
    for img_file in save_dir.glob('*.png'):
        logger.info(f"  - {img_file.name}")


if __name__ == "__main__":
    main()
