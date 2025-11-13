"""
Evaluation script for H4: Training Stability

Compares training dynamics between hierarchical and flat actor-critic:
learning curve smoothness, update magnitudes, convergence speed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import logging
from experiments.shared.evaluation_framework import HypothesisEvaluator

logger = logging.getLogger(__name__)


def compute_h4_metrics(hierarchical_data, flat_data):
    """
    Compute H4-specific metrics comparing training stability.
    
    Args:
        hierarchical_data: Metrics from hierarchical (baseline) experiment
        flat_data: Metrics from flat (H1) experiment
    """
    import torch
    import torch.nn.functional as F
    
    metrics = {
        'hierarchical': {},
        'flat': {},
        'comparison': {},
    }
    
    # Extract learning curves
    h_returns = hierarchical_data.get('episode_returns', [])
    f_returns = flat_data.get('episode_returns', [])
    
    # Learning curve smoothness (variance of differences)
    if len(h_returns) > 1:
        h_diffs = np.diff(h_returns)
        metrics['hierarchical']['curve_smoothness'] = float(np.std(h_diffs))
        metrics['hierarchical']['curve_variance'] = float(np.var(h_returns))
        # Count spikes (large positive or negative changes)
        spike_threshold = 2.0 * np.std(h_diffs)
        metrics['hierarchical']['spike_count'] = int(np.sum(np.abs(h_diffs) > spike_threshold))
        metrics['hierarchical']['spike_rate'] = float(metrics['hierarchical']['spike_count'] / len(h_diffs))
    else:
        metrics['hierarchical']['curve_smoothness'] = 0.0
        metrics['hierarchical']['curve_variance'] = 0.0
        metrics['hierarchical']['spike_count'] = 0
        metrics['hierarchical']['spike_rate'] = 0.0
    
    if len(f_returns) > 1:
        f_diffs = np.diff(f_returns)
        metrics['flat']['curve_smoothness'] = float(np.std(f_diffs))
        metrics['flat']['curve_variance'] = float(np.var(f_returns))
        spike_threshold = 2.0 * np.std(f_diffs)
        metrics['flat']['spike_count'] = int(np.sum(np.abs(f_diffs) > spike_threshold))
        metrics['flat']['spike_rate'] = float(metrics['flat']['spike_count'] / len(f_diffs))
    else:
        metrics['flat']['curve_smoothness'] = 0.0
        metrics['flat']['curve_variance'] = 0.0
        metrics['flat']['spike_count'] = 0
        metrics['flat']['spike_rate'] = 0.0
    
    # Time to target (episodes to reach 80% of max return)
    if h_returns:
        h_max = np.max(h_returns)
        h_target = 0.8 * h_max
        h_time_to_target = next((i for i, r in enumerate(h_returns) if r >= h_target), len(h_returns))
        metrics['hierarchical']['time_to_target'] = int(h_time_to_target)
    else:
        metrics['hierarchical']['time_to_target'] = len(h_returns)
    
    if f_returns:
        f_max = np.max(f_returns)
        f_target = 0.8 * f_max
        f_time_to_target = next((i for i, r in enumerate(f_returns) if r >= f_target), len(f_returns))
        metrics['flat']['time_to_target'] = int(f_time_to_target)
    else:
        metrics['flat']['time_to_target'] = len(f_returns)
    
    # Comparison
    metrics['comparison']['smoothness_improvement'] = (
        metrics['flat']['curve_smoothness'] - metrics['hierarchical']['curve_smoothness']
    )
    metrics['comparison']['convergence_speedup'] = (
        metrics['flat']['time_to_target'] - metrics['hierarchical']['time_to_target']
    )
    metrics['comparison']['spike_reduction'] = (
        metrics['flat']['spike_rate'] - metrics['hierarchical']['spike_rate']
    )
    
    # Extract training stats if available
    h_policy_losses = hierarchical_data.get('policy_losses', [])
    f_policy_losses = flat_data.get('policy_losses', [])
    h_value_losses = hierarchical_data.get('value_losses', [])
    f_value_losses = flat_data.get('value_losses', [])
    
    # Update magnitude (L2-norm approximation from loss magnitude)
    if h_policy_losses:
        metrics['hierarchical']['mean_policy_loss'] = float(np.mean(np.abs(h_policy_losses)))
        metrics['hierarchical']['std_policy_loss'] = float(np.std(h_policy_losses))
        # Approximate L2-norm from loss (loss is proportional to gradient)
        metrics['hierarchical']['mean_update_magnitude'] = float(np.mean(np.abs(h_policy_losses)))
    if f_policy_losses:
        metrics['flat']['mean_policy_loss'] = float(np.mean(np.abs(f_policy_losses)))
        metrics['flat']['std_policy_loss'] = float(np.std(f_policy_losses))
        metrics['flat']['mean_update_magnitude'] = float(np.mean(np.abs(f_policy_losses)))
    
    # Value loss magnitude
    if h_value_losses:
        metrics['hierarchical']['mean_value_loss'] = float(np.mean(np.abs(h_value_losses)))
    if f_value_losses:
        metrics['flat']['mean_value_loss'] = float(np.mean(np.abs(f_value_losses)))
    
    # KL divergence approximation (from policy entropy changes)
    h_entropies = hierarchical_data.get('entropies', [])
    f_entropies = flat_data.get('entropies', [])
    
    if len(h_entropies) > 1:
        # Approximate KL from entropy changes (entropy drop ≈ policy change)
        h_entropy_diffs = np.abs(np.diff(h_entropies))
        metrics['hierarchical']['mean_kl_approximation'] = float(np.mean(h_entropy_diffs))
    if len(f_entropies) > 1:
        f_entropy_diffs = np.abs(np.diff(f_entropies))
        metrics['flat']['mean_kl_approximation'] = float(np.mean(f_entropy_diffs))
    
    # Option duration sanity (median τ per option) - from hierarchical only
    h_option_durations = hierarchical_data.get('option_durations', {})
    if h_option_durations:
        option_medians = {}
        for opt, durations in h_option_durations.items():
            if durations:
                option_medians[opt] = {
                    'median': float(np.median(durations)),
                    'mean': float(np.mean(durations)),
                    'q25': float(np.percentile(durations, 25)),
                    'q75': float(np.percentile(durations, 75)),
                }
        metrics['hierarchical']['option_duration_sanity'] = option_medians
    
    # Final performance variance (last 100 episodes)
    if len(h_returns) >= 100:
        metrics['hierarchical']['final_performance_variance'] = float(np.var(h_returns[-100:]))
        metrics['hierarchical']['final_performance_mean'] = float(np.mean(h_returns[-100:]))
    if len(f_returns) >= 100:
        metrics['flat']['final_performance_variance'] = float(np.var(f_returns[-100:]))
        metrics['flat']['final_performance_mean'] = float(np.mean(f_returns[-100:]))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate H4: Training Stability')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Path to baseline (hierarchical) experiment')
    parser.add_argument('--flat-dir', type=str, required=True,
                       help='Path to flat (H1) experiment')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    h_dir = Path(args.baseline_dir)
    f_dir = Path(args.flat_dir)
    
    if not h_dir.exists() or not f_dir.exists():
        print("❌ Experiment directories not found")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else Path('experiments/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("H4 EVALUATION: TRAINING STABILITY")
    print("=" * 80)
    
    # Load both experiments
    h_eval = HypothesisEvaluator('baseline', output_dir)
    h_eval.load_experiment_data(h_dir)
    
    f_eval = HypothesisEvaluator('h1', output_dir)
    f_eval.load_experiment_data(f_dir)
    
    # Compute H4 metrics
    h4_metrics = compute_h4_metrics(h_eval.metrics, f_eval.metrics)
    
    # Save
    all_metrics = {
        'experiment_name': 'h4',
        'hypothesis': 'H4',
        'h4_metrics': h4_metrics,
    }
    
    # Save to major_results if applicable
    try:
        from src.utils.major_results_manager import MajorResultsManager
        manager = MajorResultsManager()
        exp_dir = manager.get_latest_experiment('baseline')  # Use baseline as reference
        if exp_dir:
            eval_dir = exp_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            output_file = eval_dir / 'h4_metrics.json'
            with open(output_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # Generate plots
            from experiments.shared.evaluation_visualizations import generate_hypothesis_plots
            viz_dir = exp_dir / "visualizations" / "evaluation"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Load baseline and H1 metrics for comparison
            baseline_metrics = None
            h1_metrics = None
            baseline_h1_file = h_dir / "evaluation" / "h1_metrics.json"
            h1_h1_file = f_dir / "evaluation" / "h1_metrics.json"
            
            if baseline_h1_file.exists():
                with open(baseline_h1_file, 'r') as f:
                    baseline_metrics = json.load(f)
            if h1_h1_file.exists():
                with open(h1_h1_file, 'r') as f:
                    h1_metrics = json.load(f)
            
            # Load training logs
            def load_training_logs(exp_dir: Path) -> dict:
                """Load training logs from experiment directory."""
                try:
                    logs = {}
                    logs_dir = exp_dir / "training" / "logs"
                    
                    # Load learning curves
                    learning_curves_files = list(logs_dir.glob("learning_curves_*.json"))
                    if learning_curves_files:
                        with open(learning_curves_files[0], 'r') as f:
                            learning_curves = json.load(f)
                            logs['episode_returns'] = learning_curves.get('episode_returns', [])
                    
                    # Load RL metrics for update norms
                    rl_metrics_files = list(logs_dir.glob("rl_metrics_*.json"))
                    if rl_metrics_files:
                        with open(rl_metrics_files[0], 'r') as f:
                            rl_metrics = json.load(f)
                            logs['update_norms'] = rl_metrics.get('update_norms', [])
                    
                    return logs
                except Exception as e:
                    logger.warning(f"Failed to load training logs: {e}")
                    return {}
            
            training_logs = {
                'baseline': load_training_logs(h_dir),
                'h1': load_training_logs(f_dir)
            }
            
            generate_hypothesis_plots(
                hypothesis='h4',
                metrics=all_metrics,
                output_dir=viz_dir,
                baseline_metrics=baseline_metrics,
                comparison_metrics=h1_metrics,
                training_logs=training_logs
            )
            
            print(f"✅ Plots generated automatically in {viz_dir}")
    except Exception as e:
        logger.warning(f"Failed to save to major_results or generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Also save locally
    output_file = output_dir / 'h4_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✅ H4 evaluation complete!")
    print(f"Hierarchical smoothness: {h4_metrics['hierarchical']['curve_smoothness']:.4f}")
    print(f"Flat smoothness: {h4_metrics['flat']['curve_smoothness']:.4f}")
    print(f"Smoothness improvement: {h4_metrics['comparison']['smoothness_improvement']:.4f}")


if __name__ == '__main__':
    main()

