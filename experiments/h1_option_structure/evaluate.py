"""
Evaluation script for H1 experiment: Option Structure

Computes H1-specific metrics comparing hierarchical vs flat policies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from experiments.shared.evaluation_framework import HypothesisEvaluator


def compute_h1_metrics(turn_data, episode_data):
    """Compute H1-specific metrics: option coherence, switch rates."""
    metrics = {
        'coherent_span_lengths': [],
        'switch_rate_per_100_turns': 0.0,
        'option_persistence': {},
    }
    
    # Track option sequences
    current_option = None
    span_length = 0
    total_switches = 0
    total_turns = len(turn_data)
    
    for turn in turn_data:
        option = turn.get('option', 'Unknown')
        
        if option != current_option:
            if current_option is not None:
                # Option switch occurred
                total_switches += 1
                if span_length > 0:
                    metrics['coherent_span_lengths'].append(span_length)
                    if current_option not in metrics['option_persistence']:
                        metrics['option_persistence'][current_option] = []
                    metrics['option_persistence'][current_option].append(span_length)
            current_option = option
            span_length = 1
        else:
            span_length += 1
    
    # Final span
    if span_length > 0 and current_option:
        metrics['coherent_span_lengths'].append(span_length)
        if current_option not in metrics['option_persistence']:
            metrics['option_persistence'][current_option] = []
        metrics['option_persistence'][current_option].append(span_length)
    
    # Compute switch rate
    if total_turns > 0:
        metrics['switch_rate_per_100_turns'] = (total_switches / total_turns) * 100
    
    # Average span lengths
    if metrics['coherent_span_lengths']:
        metrics['mean_coherent_span'] = np.mean(metrics['coherent_span_lengths'])
        metrics['median_coherent_span'] = np.median(metrics['coherent_span_lengths'])
    else:
        metrics['mean_coherent_span'] = 0.0
        metrics['median_coherent_span'] = 0.0
    
    # Average persistence per option
    metrics['mean_persistence_per_option'] = {
        opt: np.mean(lengths) if lengths else 0.0
        for opt, lengths in metrics['option_persistence'].items()
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate H1: Option Structure')
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("H1 EVALUATION: OPTION STRUCTURE")
    print("=" * 80)
    
    # Load data
    evaluator = HypothesisEvaluator('h1', output_dir)
    evaluator.load_experiment_data(exp_dir)
    
    # Compute and save metrics (automatically generates plots)
    metrics = evaluator.save_metrics('h1')
    
    print(f"\n✅ H1 evaluation complete!")
    h1_metrics = metrics.get('h1_metrics', {})
    common_metrics = metrics.get('common_metrics', {})
    print(f"Mean coherent span: {h1_metrics.get('mean_coherent_span', 0.0):.2f} turns")
    print(f"Switch rate: {h1_metrics.get('switch_rate_per_100_turns', 0.0):.2f} per 100 turns")
    print(f"Mean return: {common_metrics.get('mean_return', 0.0):.3f}")
    print(f"✅ Plots generated automatically in visualizations/evaluation/")


if __name__ == '__main__':
    main()

