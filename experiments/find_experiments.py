"""
Helper script to find and list all experiment directories.

Usage:
    python experiments/find_experiments.py
    python experiments/find_experiments.py --save-paths
"""

import argparse
import glob
import json
from pathlib import Path
from datetime import datetime


def find_experiments():
    """Find all experiment directories."""
    experiments = {}
    
    # Baseline
    baseline_pattern = 'training_logs/experiments/*/exp_*_baseline_*'
    baseline_dirs = sorted(glob.glob(baseline_pattern))
    if baseline_dirs:
        experiments['baseline'] = baseline_dirs[-1]  # Most recent
    
    # Hypothesis experiments
    hypothesis_patterns = {
        'h1': 'training_logs/experiments/*/major_*_h1_flat_policy_*',
        'h2': 'training_logs/experiments/*/major_*_h2_fixed_duration_*',
        'h3': 'training_logs/experiments/*/major_*_h3_minimal_prompts_*',
        'h5': 'training_logs/experiments/*/major_*_h5_state_ablation_*',
        'h6': 'training_logs/experiments/*/major_*_h6_transition_reward_*',
    }
    
    for h, pattern in hypothesis_patterns.items():
        dirs = sorted(glob.glob(pattern))
        if dirs:
            experiments[h] = dirs[-1]  # Most recent
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Find experiment directories')
    parser.add_argument('--save-paths', action='store_true',
                       help='Save paths to experiment_paths.json')
    parser.add_argument('--format', type=str, default='human',
                       choices=['human', 'json', 'bash'],
                       help='Output format')
    
    args = parser.parse_args()
    
    experiments = find_experiments()
    
    if args.format == 'human':
        print("=" * 80)
        print("EXPERIMENT DIRECTORY PATHS")
        print("=" * 80)
        print()
        
        for name, path in experiments.items():
            if path:
                print(f"✅ {name.upper()}:")
                print(f"   {path}")
                
                # Check for key files
                path_obj = Path(path)
                if (path_obj / 'logs' / 'metrics_tracker_*.json').exists():
                    metrics_files = list((path_obj / 'logs').glob('metrics_tracker_*.json'))
                    if metrics_files:
                        print(f"   📊 Has metrics: {metrics_files[0].name}")
                
                if (path_obj / 'models' / 'trained_agent.pt').exists():
                    print(f"   🤖 Has trained model")
                
                print()
            else:
                print(f"❌ {name.upper()}: NOT FOUND")
                print()
        
        print("=" * 80)
        
        if not all(experiments.values()):
            missing = [k for k, v in experiments.items() if not v]
            print(f"\n⚠️  Missing experiments: {', '.join(missing)}")
    
    elif args.format == 'json':
        print(json.dumps(experiments, indent=2))
    
    elif args.format == 'bash':
        for name, path in experiments.items():
            if path:
                print(f'export {name.upper()}_DIR="{path}"')
    
    if args.save_paths:
        output_file = Path('experiments/experiment_paths.json')
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'experiments': experiments
            }, f, indent=2)
        print(f"\n✅ Saved paths to {output_file}")


if __name__ == '__main__':
    main()

