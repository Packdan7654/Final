"""
Verify and Generate All Plots for All Variations

This script:
1. Checks training logs and experiments for a given date
2. Verifies all required metrics and plots exist
3. Generates missing plots for each variation
4. For flat RL (H1), generates action-based plots instead of option-based plots
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RL_plots.generate_rl_plots import RLPlotGenerator
from tools.create_evaluation_plots import HRLEvaluationPlotter


def detect_experiment_type(experiment_dir: Path) -> str:
    """
    Detect if experiment is flat RL, hierarchical RL, or other variant.
    
    Returns: 'flat', 'hierarchical', or variant name
    """
    # Check metadata
    metadata_file = experiment_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            exp_name = metadata.get('experiment_name', '').lower()
            if 'flat' in exp_name or 'h1' in exp_name:
                return 'flat'
    
    # Check experiment directory name
    exp_name = experiment_dir.name.lower()
    if 'flat' in exp_name or 'h1' in exp_name:
        return 'flat'
    elif 'h3' in exp_name:
        return 'h3'
    elif 'h5' in exp_name:
        return 'h5'
    elif 'h6' in exp_name:
        return 'h6'
    elif 'h7' in exp_name:
        return 'h7'
    else:
        return 'hierarchical'


def check_required_files(experiment_dir: Path) -> Dict[str, bool]:
    """Check if all required files exist."""
    checks = {
        'metrics_tracker': len(list((experiment_dir / 'logs').glob('metrics_tracker_*.json'))) > 0,
        'rl_plots_dir': (experiment_dir / 'RL_plots').exists(),
        'evaluation_dir': (experiment_dir / 'evaluation').exists(),
        'metadata': (experiment_dir / 'metadata.json').exists(),
    }
    return checks


def generate_rl_plots(experiment_dir: Path, exp_type: str):
    """Generate RL plots, handling flat RL specially."""
    try:
        print(f"\n  Generating RL plots...")
        generator = RLPlotGenerator(str(experiment_dir))
        
        # For flat RL, we need to modify action distribution plot
        if exp_type == 'flat':
            # Temporarily modify plot_action_distribution to use flat actions
            original_plot = generator.plot_action_distribution
            
            def plot_flat_action_distribution():
                """Plot flat action distribution instead of options."""
                # Try to load flat action counts from metrics
                metrics = generator.metrics
                
                # Check if we have flat_action_counts in metrics
                # If not, try to reconstruct from training history
                flat_action_counts = {}
                
                # Try to get from metrics_tracker if available
                if 'flat_action_counts' in metrics:
                    flat_action_counts = metrics['flat_action_counts']
                else:
                    # Try to reconstruct from episode_option_usage
                    # For flat RL, each "option" is actually a flat action
                    option_counts = metrics.get('option_counts', {})
                    if option_counts:
                        # In flat RL, option_counts might contain flat action names
                        flat_action_counts = option_counts
                
                if not flat_action_counts:
                    print("[!] No flat action usage data")
                    return
                
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Action distribution
                actions = list(flat_action_counts.keys())
                counts = list(flat_action_counts.values())
                total = sum(counts) if counts else 1
                percentages = [c/total*100 for c in counts]
                
                colors = sns.color_palette("Set2", len(actions))
                bars = ax1.bar(actions, percentages, color=colors, alpha=0.7, edgecolor='black')
                ax1.set_ylabel('Usage Percentage (%)', fontsize=12)
                ax1.set_xlabel('Flat Action', fontsize=12)
                ax1.set_title('Flat Action Usage Distribution', fontweight='bold', pad=15)
                ax1.grid(True, alpha=0.3, axis='y')
                ax1.tick_params(axis='x', rotation=45, ha='right')
                
                # Add percentage labels
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
                
                # Pie chart
                ax2.pie(percentages, labels=actions, autopct='%1.1f%%', colors=colors, startangle=90)
                ax2.set_title('Flat Action Usage Proportion', fontweight='bold', pad=15)
                
                plt.tight_layout()
                save_path = generator.output_dir / '13_action_distribution.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[+] Flat action distribution -> {save_path.name}")
            
            generator.plot_action_distribution = plot_flat_action_distribution
        
        generator.generate_all_plots()
        print(f"  [OK] RL plots generated")
        return True
    except Exception as e:
        print(f"  [FAILED] Failed to generate RL plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_evaluation_plots(experiment_dir: Path):
    """Generate evaluation plots."""
    try:
        print(f"\n  Generating evaluation plots...")
        plotter = HRLEvaluationPlotter(str(experiment_dir))
        plotter.load_data()
        plotter.generate_all_plots()
        print(f"  [OK] Evaluation plots generated")
        return True
    except Exception as e:
        print(f"  [FAILED] Failed to generate evaluation plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_experiment(experiment_dir: Path, generate_missing: bool = True):
    """Process a single experiment."""
    exp_name = experiment_dir.name
    print(f"\n{'='*80}")
    print(f"Processing: {exp_name}")
    print(f"{'='*80}")
    
    # Detect experiment type
    exp_type = detect_experiment_type(experiment_dir)
    print(f"  Type: {exp_type}")
    
    # Check required files
    checks = check_required_files(experiment_dir)
    print(f"  File checks:")
    for name, exists in checks.items():
        status = "[OK]" if exists else "[MISSING]"
        print(f"    {status} {name}")
    
    # Generate plots if requested
    if generate_missing:
        if not checks['rl_plots_dir'] or len(list((experiment_dir / 'RL_plots').glob('*.png'))) < 5:
            generate_rl_plots(experiment_dir, exp_type)
        else:
            print(f"  [OK] RL plots already exist")
        
        if not checks['evaluation_dir'] or len(list((experiment_dir / 'evaluation').rglob('*.png'))) < 5:
            generate_evaluation_plots(experiment_dir)
        else:
            print(f"  [OK] Evaluation plots already exist")
    
    return checks


def main():
    parser = argparse.ArgumentParser(
        description='Verify and generate all plots for experiments'
    )
    parser.add_argument('--date', type=str, default='20251111',
                       help='Date to check (YYYYMMDD format)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment directory name')
    parser.add_argument('--generate', action='store_true',
                       help='Generate missing plots')
    parser.add_argument('--all-dates', action='store_true',
                       help='Process all dates in training_logs/experiments/')
    
    args = parser.parse_args()
    
    experiments_dir = project_root / 'training_logs' / 'experiments'
    
    if args.experiment:
        # Process specific experiment
        exp_path = experiments_dir / args.date / args.experiment
        if not exp_path.exists():
            print(f"[ERROR] Experiment not found: {exp_path}")
            return
        process_experiment(exp_path, generate_missing=args.generate)
    elif args.all_dates:
        # Process all dates
        dates = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        for date_dir in dates:
            print(f"\n\n{'#'*80}")
            print(f"Processing date: {date_dir.name}")
            print(f"{'#'*80}")
            experiments = sorted([e for e in date_dir.iterdir() if e.is_dir()])
            for exp_dir in experiments:
                process_experiment(exp_dir, generate_missing=args.generate)
    else:
        # Process specific date
        date_dir = experiments_dir / args.date
        if not date_dir.exists():
            print(f"[ERROR] Date directory not found: {date_dir}")
            return
        
        experiments = sorted([e for e in date_dir.iterdir() if e.is_dir()])
        print(f"Found {len(experiments)} experiments for {args.date}")
        
        for exp_dir in experiments:
            process_experiment(exp_dir, generate_missing=args.generate)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

