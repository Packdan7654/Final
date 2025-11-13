"""
Evaluation script for H5 experiment: State Ablation

Computes H5-specific metrics and generates comparison reports.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from experiments.shared.evaluation_framework import HypothesisEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate H5: State Ablation')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiment results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("H5 EVALUATION: STATE ABLATION")
    print("=" * 80)
    print(f"Experiment: {exp_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()
    
    # Initialize evaluator
    evaluator = HypothesisEvaluator('h5', output_dir)
    evaluator.load_experiment_data(exp_dir)
    
    # Compute and save metrics
    metrics = evaluator.save_metrics('h5')
    
    print("✅ H5 evaluation complete!")
    print(f"\nMetrics saved to: {output_dir / 'h5_metrics.json'}")
    print(f"\nState Dimension: {metrics.get('h5_metrics', {}).get('state_dimension', 'N/A')}")
    print(f"Compression Ratio: {metrics.get('h5_metrics', {}).get('state_compression_ratio', 0.0):.2%}")
    print(f"Mean Return: {metrics.get('common_metrics', {}).get('mean_return', 0.0):.3f}")
    print(f"✅ Plots generated automatically in visualizations/evaluation/")


if __name__ == '__main__':
    main()

