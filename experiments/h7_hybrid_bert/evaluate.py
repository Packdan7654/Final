"""
Evaluation script for H7 experiment: Hybrid BERT

Computes H7-specific metrics:
- Dialogue coherence score (reference resolution accuracy)
- Context-dependent question answering rate
- Turn-aware response appropriateness
- Contradiction rate
- Embedding similarity between consecutive turns
- Multi-turn reference tracking accuracy
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from experiments.shared.evaluation_framework import HypothesisEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate H7: Hybrid BERT')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to H7 experiment results directory')
    parser.add_argument('--baseline-dir', type=str, default=None,
                       help='Path to baseline experiment directory for comparison')
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
    print("H7 EVALUATION: HYBRID BERT")
    print("=" * 80)
    print(f"Experiment: {exp_dir}")
    if args.baseline_dir:
        print(f"Baseline: {args.baseline_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print()
    
    # Initialize evaluator
    evaluator = HypothesisEvaluator('h7', output_dir)
    evaluator.load_experiment_data(exp_dir)
    
    # Compute and save metrics
    metrics = evaluator.save_metrics('h7')
    
    print("✅ H7 evaluation complete!")
    print(f"\nMetrics saved to: {output_dir / 'h7_metrics.json'}")
    
    # Print key metrics
    h7_metrics = metrics.get('h7_metrics', {})
    print(f"\nH7-Specific Metrics:")
    print(f"  Dialogue Coherence: {h7_metrics.get('dialogue_coherence', 0.0):.3f}")
    print(f"  Context-Dependent QA Rate: {h7_metrics.get('context_dependent_qa_rate', 0.0):.3f}")
    print(f"  Contradiction Rate: {h7_metrics.get('contradiction_rate', 0.0):.3f}")
    print(f"  Embedding Similarity: {h7_metrics.get('embedding_similarity', 0.0):.3f}")
    
    common_metrics = metrics.get('common_metrics', {})
    print(f"\nCommon Metrics:")
    print(f"  Mean Return: {common_metrics.get('mean_return', 0.0):.3f}")
    print(f"  Mean Coverage: {common_metrics.get('mean_coverage', 0.0):.3f}")
    print(f"\n✅ Plots generated automatically in visualizations/evaluation/")


if __name__ == '__main__':
    main()

