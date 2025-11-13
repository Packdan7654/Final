"""
Major Results Plotter

Specialized plotter for generating publication-ready visualizations
from major_results/ directory structure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class MajorResultsPlotter:
    """
    Generates visualizations from major_results/ directory.
    
    Loads data from metrics/ folder and generates publication-ready plots.
    """
    
    def __init__(self, model_dir: Path):
        """
        Initialize plotter for a model.
        
        Args:
            model_dir: Path to model directory in major_results/
        """
        self.model_dir = Path(model_dir)
        self.metrics_dir = self.model_dir / "metrics"
        self.viz_dir = self.model_dir / "visualizations"
        
        # Load metrics
        self.training_metrics = {}
        self.rl_metrics = {}
        self.learning_curves = {}
        self.convergence_report = {}
        
        self._load_metrics()
    
    def _load_metrics(self):
        """Load all metrics from metrics/ directory."""
        # Load training metrics
        training_metrics_file = self.metrics_dir / "training_metrics.json"
        if training_metrics_file.exists():
            with open(training_metrics_file, 'r') as f:
                self.training_metrics = json.load(f)
        
        # Load RL metrics
        rl_metrics_file = self.metrics_dir / "rl_metrics.json"
        if rl_metrics_file.exists():
            with open(rl_metrics_file, 'r') as f:
                self.rl_metrics = json.load(f)
        
        # Load learning curves
        learning_curves_file = self.metrics_dir / "learning_curves.json"
        if learning_curves_file.exists():
            with open(learning_curves_file, 'r') as f:
                self.learning_curves = json.load(f)
        
        # Load convergence report
        convergence_file = self.metrics_dir / "convergence_report.json"
        if convergence_file.exists():
            with open(convergence_file, 'r') as f:
                self.convergence_report = json.load(f)
    
    def plot_learning_curve_advanced(self, output_dir: Optional[Path] = None):
        """Generate advanced learning curve with multiple analysis views."""
        if output_dir is None:
            output_dir = self.viz_dir / "advanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        returns = np.array(self.learning_curves.get('episode_returns', []))
        if len(returns) == 0:
            logger.warning("No episode returns data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = np.arange(1, len(returns) + 1)
        
        # Top-left: Learning curve with moving averages
        ax = axes[0, 0]
        ax.plot(episodes, returns, alpha=0.2, color='steelblue', linewidth=0.5)
        
        # Multiple moving averages
        for window in [25, 50, 100]:
            if len(returns) >= window:
                ma = np.convolve(returns, np.ones(window)/window, mode='valid')
                ma_episodes = np.arange(window, len(returns) + 1)
                ax.plot(ma_episodes, ma, linewidth=2, label=f'{window}-ep MA')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Learning Curve with Moving Averages')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top-right: Return distribution
        ax = axes[0, 1]
        ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Learning rate (improvement over time)
        ax = axes[1, 0]
        if len(returns) >= 100:
            window = 50
            improvements = []
            for i in range(window, len(returns)):
                early = np.mean(returns[i-window:i-window//2])
                late = np.mean(returns[i-window//2:i])
                improvements.append(late - early)
            ax.plot(episodes[window:], improvements, linewidth=2, color='green')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Improvement (50-ep window)')
            ax.set_title('Learning Rate Over Time')
            ax.grid(True, alpha=0.3)
        
        # Bottom-right: Stability analysis
        ax = axes[1, 1]
        if len(returns) >= 100:
            window = 50
            stds = []
            for i in range(window-1, len(returns)):
                stds.append(np.std(returns[i-window+1:i+1]))
            ax.plot(episodes[window-1:], stds, linewidth=2, color='orange')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Std Dev (50-ep window)')
            ax.set_title('Learning Stability')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_curve_advanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved advanced learning curve to {output_dir / 'learning_curve_advanced.png'}")
    
    def plot_rl_metrics_advanced(self, output_dir: Optional[Path] = None):
        """Generate advanced RL metrics plots."""
        if output_dir is None:
            output_dir = self.viz_dir / "advanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Value and policy losses
        if self.rl_metrics.get('learning_dynamics', {}).get('value_losses'):
            value_losses = self.rl_metrics['learning_dynamics']['value_losses']
            axes[0, 0].plot(value_losses, alpha=0.7, color='blue')
            axes[0, 0].set_title('Value Loss Over Training')
            axes[0, 0].set_xlabel('Update')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.rl_metrics.get('learning_dynamics', {}).get('policy_losses'):
            policy_losses = self.rl_metrics['learning_dynamics']['policy_losses']
            axes[0, 1].plot(policy_losses, alpha=0.7, color='red')
            axes[0, 1].set_title('Policy Loss Over Training')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient and update norms
        if self.rl_metrics.get('learning_dynamics', {}).get('gradient_norms'):
            grad_norms = self.rl_metrics['learning_dynamics']['gradient_norms']
            axes[1, 0].plot(grad_norms, alpha=0.7, color='green')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('L2 Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.rl_metrics.get('learning_dynamics', {}).get('update_norms'):
            update_norms = self.rl_metrics['learning_dynamics']['update_norms']
            axes[1, 1].plot(update_norms, alpha=0.7, color='purple')
            axes[1, 1].set_title('Parameter Update Norms')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('L2 Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Entropy and value estimates
        if self.rl_metrics.get('policy_learning', {}).get('entropy_over_time'):
            entropies = self.rl_metrics['policy_learning']['entropy_over_time']
            axes[2, 0].plot(entropies, alpha=0.7, color='orange')
            axes[2, 0].set_title('Policy Entropy')
            axes[2, 0].set_xlabel('Update')
            axes[2, 0].set_ylabel('Entropy')
            axes[2, 0].grid(True, alpha=0.3)
        
        if self.rl_metrics.get('value_learning', {}).get('value_estimates'):
            value_ests = self.rl_metrics['value_learning']['value_estimates']
            axes[2, 1].plot(value_ests, alpha=0.7, color='cyan')
            axes[2, 1].set_title('Value Estimates')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Value')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rl_metrics_advanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved advanced RL metrics to {output_dir / 'rl_metrics_advanced.png'}")
    
    def plot_convergence_analysis_advanced(self, output_dir: Optional[Path] = None):
        """Generate advanced convergence analysis."""
        if output_dir is None:
            output_dir = self.viz_dir / "advanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        returns = np.array(self.learning_curves.get('episode_returns', []))
        if len(returns) == 0:
            return
        
        conv_ep = self.convergence_report.get('episode')
        if conv_ep is None:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = np.arange(1, len(returns) + 1)
        
        # Top: Learning curve with convergence point
        ax = axes[0]
        ax.plot(episodes, returns, alpha=0.3, color='steelblue', linewidth=0.5)
        
        # Mark convergence
        if conv_ep < len(returns):
            ax.axvline(x=conv_ep, color='red', linestyle='--', linewidth=2, label=f'Convergence (ep {conv_ep})')
            ax.scatter([conv_ep], [returns[conv_ep-1]], color='red', s=200, zorder=5, marker='*')
        
        # Pre and post convergence windows
        window = 50
        if conv_ep and conv_ep >= window and conv_ep + window <= len(returns):
            pre_window = returns[conv_ep-window:conv_ep]
            post_window = returns[conv_ep:conv_ep+window]
            ax.axvspan(conv_ep-window, conv_ep, alpha=0.2, color='yellow', label='Pre-convergence')
            ax.axvspan(conv_ep, conv_ep+window, alpha=0.2, color='green', label='Post-convergence')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom: Convergence metrics
        ax = axes[1]
        conv_data = {
            'Episode': conv_ep,
            'Samples': self.convergence_report.get('samples', 0),
            'Time (s)': self.convergence_report.get('time_seconds', 0)
        }
        if conv_data['Time (s)'] > 0:
            conv_data['Time (h)'] = conv_data['Time (s)'] / 3600
        
        # Create text summary
        text = "Convergence Metrics:\n" + "="*30 + "\n"
        for key, value in conv_data.items():
            if isinstance(value, float):
                text += f"{key}: {value:.2f}\n"
            else:
                text += f"{key}: {value}\n"
        
        ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'convergence_analysis_advanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved advanced convergence analysis to {output_dir / 'convergence_analysis_advanced.png'}")
    
    def generate_all_advanced_plots(self):
        """Generate all advanced visualizations."""
        advanced_dir = self.viz_dir / "advanced"
        advanced_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating advanced visualizations for {self.model_dir.name}...")
        
        self.plot_learning_curve_advanced(advanced_dir)
        self.plot_rl_metrics_advanced(advanced_dir)
        self.plot_convergence_analysis_advanced(advanced_dir)
        
        print(f"✅ Advanced visualizations saved to {advanced_dir}")

