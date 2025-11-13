"""
Verify Evaluation Outputs

Checks that all required plots and metrics are generated for experiments.
"""

from pathlib import Path
import json

def verify_experiment(exp_dir: Path):
    """Verify all outputs for an experiment."""
    results = {
        'name': exp_dir.name,
        'evaluation_plots': 0,
        'rl_plots': 0,
        'metrics': False,
        'evaluation_metrics': False
    }
    
    # Check evaluation plots
    eval_dir = exp_dir / 'evaluation'
    if eval_dir.exists():
        results['evaluation_plots'] = len(list(eval_dir.rglob('*.png')))
    
    # Check RL plots
    rl_plots_dir = exp_dir / 'RL_plots'
    if rl_plots_dir.exists():
        results['rl_plots'] = len(list(rl_plots_dir.glob('*.png')))
    
    # Check training metrics
    logs_dir = exp_dir / 'logs'
    if logs_dir.exists():
        metrics_files = list(logs_dir.glob('metrics_tracker_*.json'))
        results['metrics'] = len(metrics_files) > 0
    
    # Check evaluation metrics
    # Look in experiments/results/
    from pathlib import Path as P
    project_root = P(__file__).parent.parent
    results_dir = project_root / 'experiments' / 'results'
    if results_dir.exists():
        # Check for any metrics JSON files that might match this experiment
        eval_metrics = list(results_dir.rglob('*_metrics.json'))
        results['evaluation_metrics'] = len(eval_metrics) > 0
    
    return results

def main():
    """Check experiments for a given date."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Verify evaluation outputs for experiments')
    parser.add_argument('--date', type=str, default=None,
                       help='Date to check (YYYYMMDD format, default: today)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment directory path')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.experiment:
        # Check specific experiment
        exp_path = Path(args.experiment)
        if not exp_path.exists():
            print(f"Experiment directory not found: {exp_path}")
            return
        experiments = [exp_path]
        date_str = args.experiment
    else:
        # Check all experiments for a date
        if args.date:
            date_str = args.date
        else:
            date_str = datetime.now().strftime("%Y%m%d")
        
        experiments_dir = project_root / 'training_logs' / 'experiments' / date_str
        
        if not experiments_dir.exists():
            print(f"Directory not found: {experiments_dir}")
            return
        
        experiments = [d for d in experiments_dir.iterdir()
                       if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n{'='*80}")
    print(f"EVALUATION OUTPUT VERIFICATION - {date_str} EXPERIMENTS")
    print(f"{'='*80}\n")
    
    for exp_dir in sorted(experiments):
        results = verify_experiment(exp_dir)
        print(f"{results['name']}:")
        print(f"  Evaluation Plots: {results['evaluation_plots']} files")
        print(f"  RL Plots: {results['rl_plots']} files")
        print(f"  Training Metrics: {'[OK]' if results['metrics'] else '[MISSING]'}")
        print(f"  Evaluation Metrics: {'[OK]' if results['evaluation_metrics'] else '[MISSING]'}")
        print()
    
    print(f"{'='*80}")
    print(f"Total experiments checked: {len(experiments)}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    import sys
    main()

