"""
Example usage of RL plot generator

This script demonstrates how to generate RL/HRL evaluation plots.
"""

from generate_rl_plots import RLPlotGenerator
from pathlib import Path

# Example: Generate plots for a specific experiment
if __name__ == '__main__':
    # Path to your experiment directory
    # Example: training_logs/experiments/20251105/exp_003_20251105_161942
    experiment_path = "training_logs/experiments/20251105/exp_003_20251105_161942"
    
    # Create generator
    generator = RLPlotGenerator(experiment_path, output_dir="RL_plots")
    
    # Generate all plots
    generator.generate_all_plots()
    
    # Or generate individual plots:
    # generator.plot_learning_curve()
    # generator.plot_option_transition_matrix()
    # generator.plot_policy_loss()
    # etc.
