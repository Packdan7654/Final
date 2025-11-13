"""
Evaluation script for H3 experiment: Prompt Headers

Computes H3-specific metrics comparing structured headers vs minimal prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from experiments.shared.evaluation_framework import HypothesisEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate H3: Prompt Headers')
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
    print("H3 EVALUATION: PROMPT HEADERS")
    print("=" * 80)
    
    evaluator = HypothesisEvaluator('h3', output_dir)
    evaluator.load_experiment_data(exp_dir)
    
    metrics = evaluator.save_metrics('h3')
    
    print(f"\n✅ H3 evaluation complete!")
    h3_metrics = metrics.get('h3_metrics', {})
    print(f"Novel Fact Coverage: {h3_metrics.get('novel_fact_coverage', 0.0):.2%}")
    print(f"Repetition Ratio: {h3_metrics.get('repetition_ratio', 0.0):.2%}")
    print(f"Hallucination Rate: {h3_metrics.get('hallucination_rate', 0.0):.2%}")
    print(f"✅ Plots generated automatically in visualizations/evaluation/")


if __name__ == '__main__':
    main()

