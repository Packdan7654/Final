"""
Evaluation script for H2 experiment: Learned Terminations

Computes H2-specific metrics comparing fixed-duration vs learned terminations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from experiments.shared.evaluation_framework import HypothesisEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate H2: Learned Terminations')
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
    print("H2 EVALUATION: LEARNED TERMINATIONS")
    print("=" * 80)
    
    evaluator = HypothesisEvaluator('h2', output_dir)
    evaluator.load_experiment_data(exp_dir)
    
    metrics = evaluator.save_metrics('h2')
    
    print(f"\n✅ H2 evaluation complete!")
    print(f"Mean Explain Duration: {metrics.get('h2_metrics', {}).get('mean_explain_duration', 0.0):.2f} turns")
    print(f"Dwell Correlation: {metrics.get('h2_metrics', {}).get('dwell_correlation', 0.0):.3f}")
    print(f"✅ Plots generated automatically in visualizations/evaluation/")


if __name__ == '__main__':
    main()

