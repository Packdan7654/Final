"""
Run all evaluation scripts automatically.

Usage:
    python experiments/run_all_evaluations.py
    python experiments/run_all_evaluations.py --experiment-paths experiments/experiment_paths.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_experiment_paths(path_file=None):
    """Load experiment paths from file or find them."""
    if path_file and Path(path_file).exists():
        with open(path_file) as f:
            data = json.load(f)
            return data.get('experiments', {})
    else:
        # Use find_experiments to get paths
        from find_experiments import find_experiments
        return find_experiments()


def run_evaluation(script_path, exp_dir, output_dir):
    """Run a single evaluation script."""
    cmd = [
        sys.executable,
        str(script_path),
        '--experiment-dir', str(exp_dir),
        '--output-dir', str(output_dir)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {script_path.name}")
    print(f"Experiment: {exp_dir}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script_path.name} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {script_path.name} failed")
        if result.stderr:
            print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all evaluation scripts')
    parser.add_argument('--experiment-paths', type=str, default=None,
                       help='JSON file with experiment paths')
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Skip these hypotheses (e.g., --skip h1 h2)')
    
    args = parser.parse_args()
    
    # Load experiment paths
    experiments = load_experiment_paths(args.experiment_paths)
    
    if not experiments:
        print("❌ No experiments found. Run training first.")
        return
    
    # Evaluation scripts
    evaluations = {
        'h1': {
            'script': Path('experiments/h1_option_structure/evaluate.py'),
            'exp_key': 'h1',
            'output_dir': Path('experiments/results/h1')
        },
        'h2': {
            'script': Path('experiments/h2_learned_terminations/evaluate.py'),
            'exp_key': 'h2',
            'output_dir': Path('experiments/results/h2')
        },
        'h3': {
            'script': Path('experiments/h3_prompt_headers/evaluate.py'),
            'exp_key': 'h3',
            'output_dir': Path('experiments/results/h3')
        },
        'h5': {
            'script': Path('experiments/h5_state_ablation/evaluate.py'),
            'exp_key': 'h5',
            'output_dir': Path('experiments/results/h5')
        },
        'h6': {
            'script': Path('experiments/h6_transition_reward/evaluate.py'),
            'exp_key': 'h6',
            'output_dir': Path('experiments/results/h6')
        },
        'h7': {
            'script': Path('experiments/h7_hybrid_bert/evaluate.py'),
            'exp_key': 'h7',
            'output_dir': Path('experiments/results/h7')
        },
    }
    
    results = {}
    
    # Run standard evaluations
    for h, config in evaluations.items():
        if h in args.skip:
            print(f"⏭️  Skipping {h}")
            results[h] = True
            continue
        
        exp_dir = experiments.get(config['exp_key'])
        if not exp_dir:
            print(f"⚠️  {h}: Experiment directory not found, skipping")
            results[h] = False
            continue
        
        if not config['script'].exists():
            print(f"⚠️  {h}: Evaluation script not found: {config['script']}")
            results[h] = False
            continue
        
        config['output_dir'].mkdir(parents=True, exist_ok=True)
        
        success = run_evaluation(
            config['script'],
            exp_dir,
            config['output_dir']
        )
        results[h] = success
    
    # H4 is special - needs baseline and H1
    if 'h4' not in args.skip:
        h4_script = Path('experiments/h4_training_stability/evaluate.py')
        baseline_dir = experiments.get('baseline')
        h1_dir = experiments.get('h1')
        
        if baseline_dir and h1_dir and h4_script.exists():
            output_dir = Path('experiments/results/h4')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                sys.executable,
                str(h4_script),
                '--hierarchical-dir', str(baseline_dir),
                '--flat-dir', str(h1_dir),
                '--output-dir', str(output_dir)
            ]
            
            print(f"\n{'='*80}")
            print(f"Running: H4 Training Stability")
            print(f"Hierarchical: {baseline_dir}")
            print(f"Flat: {h1_dir}")
            print(f"{'='*80}\n")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            results['h4'] = (result.returncode == 0)
            
            if results['h4']:
                print("✅ H4 evaluation completed successfully")
            else:
                print("❌ H4 evaluation failed")
                if result.stderr:
                    print(result.stderr)
        else:
            print("⚠️  H4: Missing baseline or H1 directory, skipping")
            results['h4'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    for h, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {h.upper()}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✅ All evaluations completed successfully!")
        print("\nNext steps:")
        print("1. Generate comparison report:")
        print("   python -c \"from experiments.shared.comparison_tools import HypothesisComparator; from pathlib import Path; HypothesisComparator(Path('experiments/results')).generate_comparison_report(Path('experiments/results/comparison_report.json'))\"")
        print("2. Extract metrics for thesis:")
        print("   python extract_thesis_metrics.py")
    else:
        print("\n⚠️  Some evaluations failed. Check errors above.")


if __name__ == '__main__':
    main()

