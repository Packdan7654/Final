"""
Master script to run all hypothesis experiments and generate comparisons.

Usage:
    python experiments/run_all_experiments.py --episodes 600 --device cuda
"""

import sys
from pathlib import Path
import subprocess
import argparse
from datetime import datetime


def run_experiment(script_path: Path, name: str, episodes: int, device: str, verbose: bool = False):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        str(script_path),
        '--episodes', str(episodes),
        '--name', name,
        '--device', device,
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    result = subprocess.run(cmd, capture_output=not verbose)
    
    if result.returncode == 0:
        print(f"\n✅ {name} completed successfully")
        return True
    else:
        print(f"\n❌ {name} failed with return code {result.returncode}")
        if not verbose:
            print(result.stderr.decode())
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all hypothesis experiments')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes per experiment')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline training (if already done)')
    parser.add_argument('--skip-h1', action='store_true',
                       help='Skip H1 experiment (flat policy)')
    parser.add_argument('--skip-h2', action='store_true',
                       help='Skip H2 experiment (fixed duration)')
    parser.add_argument('--skip-h3', action='store_true',
                       help='Skip H3 experiment (minimal prompts)')
    parser.add_argument('--skip-h5', action='store_true',
                       help='Skip H5 experiment (state ablation)')
    parser.add_argument('--skip-h6', action='store_true',
                       help='Skip H6 experiment (transition reward)')
    parser.add_argument('--skip-h7', action='store_true',
                       help='Skip H7 experiment (hybrid BERT)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    experiments_dir = Path(__file__).parent
    
    print("=" * 80)
    print("HYPOTHESIS TESTING: RUN ALL EXPERIMENTS")
    print("=" * 80)
    print(f"Episodes per experiment: {args.episodes}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    results = {}
    
    # 1. Baseline
    if not args.skip_baseline:
        baseline_script = Path('train.py')
        if baseline_script.exists():
            results['baseline'] = run_experiment(
                baseline_script, 'baseline', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  Baseline script not found, skipping...")
    else:
        print("⏭️  Skipping baseline (--skip-baseline)")
        results['baseline'] = True  # Assume success
    
    # 2. H1: Flat Policy
    if not getattr(args, 'skip_h1', False):
        h1_script = experiments_dir / 'h1_option_structure' / 'train.py'
        if h1_script.exists():
            results['h1'] = run_experiment(
                h1_script, 'h1_flat_policy', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H1 script not found")
            results['h1'] = False
    else:
        print("⏭️  Skipping H1")
        results['h1'] = True
    
    # 3. H2: Fixed Duration
    if not getattr(args, 'skip_h2', False):
        h2_script = experiments_dir / 'h2_learned_terminations' / 'train.py'
        if h2_script.exists():
            results['h2'] = run_experiment(
                h2_script, 'h2_fixed_duration', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H2 script not found")
            results['h2'] = False
    else:
        print("⏭️  Skipping H2")
        results['h2'] = True
    
    # 4. H3: Minimal Prompts
    if not getattr(args, 'skip_h3', False):
        h3_script = experiments_dir / 'h3_prompt_headers' / 'train.py'
        if h3_script.exists():
            results['h3'] = run_experiment(
                h3_script, 'h3_minimal_prompts', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H3 script not found")
            results['h3'] = False
    else:
        print("⏭️  Skipping H3")
        results['h3'] = True
    
    # 5. H5: State Ablation
    if not args.skip_h5:
        h5_script = experiments_dir / 'h5_state_ablation' / 'train.py'
        if h5_script.exists():
            results['h5'] = run_experiment(
                h5_script, 'h5_state_ablation', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H5 script not found")
            results['h5'] = False
    else:
        print("⏭️  Skipping H5 (--skip-h5)")
        results['h5'] = True
    
    # 6. H6: Transition Reward
    if not args.skip_h6:
        h6_script = experiments_dir / 'h6_transition_reward' / 'train.py'
        if h6_script.exists():
            results['h6'] = run_experiment(
                h6_script, 'h6_transition_reward', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H6 script not found")
            results['h6'] = False
    else:
        print("⏭️  Skipping H6 (--skip-h6)")
        results['h6'] = True
    
    # 7. H7: Hybrid BERT
    if not args.skip_h7:
        h7_script = experiments_dir / 'h7_hybrid_bert' / 'train.py'
        if h7_script.exists():
            results['h7'] = run_experiment(
                h7_script, 'h7_hybrid_bert', args.episodes, args.device, args.verbose
            )
        else:
            print("⚠️  H7 script not found")
            results['h7'] = False
    else:
        print("⏭️  Skipping H7 (--skip-h7)")
        results['h7'] = True
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:20s}: {status}")
    print("=" * 80)
    
    if all(results.values()):
        print("\n✅ All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Run evaluation scripts for each experiment")
        print("2. Generate comparison report using comparison_tools.py")
    else:
        print("\n⚠️  Some experiments failed. Check logs above.")


if __name__ == '__main__':
    main()

