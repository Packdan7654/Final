"""
Generate Major Results Visualizations

On-demand script for generating advanced visualizations for models in major_results/.
Supports generating for a single model or all models.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.major_results_plotter import MajorResultsPlotter
from src.utils.major_results_manager import MajorResultsManager


def generate_for_model(model_name: str, advanced: bool = False, exp_dir: Path = None):
    """
    Generate visualizations for a single model or specific experiment.
    
    Args:
        model_name: Model name (normalized) - used if exp_dir is None
        advanced: Whether to generate advanced plots
        exp_dir: Optional specific experiment directory
    """
    manager = MajorResultsManager()
    
    if exp_dir is None:
        # Get latest experiment for this model
        exp_dir = manager.get_latest_experiment(model_name)
        if exp_dir is None:
            print(f"❌ No experiments found for model: {model_name}")
            return False
    else:
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            print(f"❌ Experiment directory not found: {exp_dir}")
            return False
    
    print(f"\n{'='*80}")
    print(f"Generating visualizations for: {exp_dir.name}")
    print(f"{'='*80}\n")
    
    plotter = MajorResultsPlotter(exp_dir)
    
    if advanced:
        plotter.generate_all_advanced_plots()
    else:
        # Basic plots are already generated during training
        print("Basic visualizations are auto-generated during training.")
        print("Use --advanced flag to generate advanced plots.")
    
    return True


def generate_for_all(advanced: bool = False):
    """
    Generate visualizations for all experiments.
    
    Args:
        advanced: Whether to generate advanced plots
    """
    manager = MajorResultsManager()
    experiments = manager.list_all_experiments()
    
    if not experiments:
        print("❌ No experiments found in major_results/")
        return
    
    print(f"\n{'='*80}")
    print(f"Generating visualizations for {len(experiments)} experiments")
    print(f"{'='*80}\n")
    
    for exp_dir in experiments:
        try:
            generate_for_model(None, advanced=advanced, exp_dir=exp_dir)
        except Exception as e:
            print(f"❌ Failed to generate visualizations for {exp_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Completed visualization generation for all experiments")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for major_results models'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to generate visualizations for latest experiment (default: all)')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment directory name (YYYYMMDD_NNN_modelname)')
    parser.add_argument('--all', action='store_true',
                       help='Generate for all experiments')
    parser.add_argument('--advanced', action='store_true',
                       help='Generate advanced plots (basic plots are auto-generated during training)')
    
    args = parser.parse_args()
    
    if args.experiment:
        exp_path = Path('major_results') / args.experiment
        generate_for_model(None, advanced=args.advanced, exp_dir=exp_path)
    elif args.model:
        generate_for_model(args.model, advanced=args.advanced)
    elif args.all:
        generate_for_all(advanced=args.advanced)
    else:
        # Default: generate for all
        generate_for_all(advanced=args.advanced)


if __name__ == '__main__':
    main()

