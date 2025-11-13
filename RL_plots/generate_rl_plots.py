"""
RL/HRL Evaluation Plot Generator
Generates individual plots for comprehensive RL/HRL analysis

Based on standard RL evaluation practices and HRL-specific metrics.
Each plot is generated individually for easy inclusion in papers/thesis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

class RLPlotGenerator:
    """
    Generates individual RL/HRL evaluation plots.
    
    Each plot is saved separately for easy inclusion in papers.
    """
    
    def __init__(self, experiment_dir: str, output_dir: str = None):
        self.exp_dir = Path(experiment_dir)
        
        # Default output to experiment directory / RL_plots
        if output_dir is None:
            self.output_dir = self.exp_dir / 'RL_plots'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.metrics = {}
        self.training_data = {}
        self.load_data()
    
    def load_data(self):
        """Load training data from experiment directory"""
        # Load metrics tracker JSON
        logs_dir = self.exp_dir / 'logs'
        if not logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
        
        metrics_files = list(logs_dir.glob('metrics_tracker_*.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                self.metrics = json.load(f)
        
        # Check metadata to detect experiment type
        metadata_file = self.exp_dir / 'metadata.json'
        self.is_flat_rl = False
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                exp_name = metadata.get('experiment_name', '').lower()
                if 'flat' in exp_name or 'h1' in exp_name:
                    self.is_flat_rl = True
        
        # Also check directory name
        if 'flat' in self.exp_dir.name.lower() or 'h1' in self.exp_dir.name.lower():
            self.is_flat_rl = True
        
        # Training data is now in metrics_tracker JSON
        # Extract RL-specific metrics from metrics
        self.training_data = {
            'value_losses': self.metrics.get('value_losses', []),
            'policy_losses': self.metrics.get('policy_losses', []),
            'entropies': self.metrics.get('entropies', []),
            'value_estimates': self.metrics.get('value_estimates', []),
            'advantages': self.metrics.get('advantages', []),
            'termination_losses': self.metrics.get('termination_losses', [])
        }
        
        print(f"[+] Loaded data from {self.exp_dir.name}")
        print(f"    Episodes: {len(self.metrics.get('episode_returns', []))}")
        print(f"    Training updates: {len(self.training_data.get('value_losses', []))}")
        if self.is_flat_rl:
            print(f"    Type: Flat RL (will use actions instead of options)")
    
    # ========== 1. LEARNING CURVES ==========
    
    def plot_learning_curve(self, window: int = 50):
        """
        Plot 1: Learning Curve - Episode Returns Over Training
        
        Standard RL metric showing cumulative reward per episode.
        Includes moving average and confidence intervals.
        """
        returns = np.array(self.metrics.get('episode_returns', []))
        if len(returns) == 0:
            print("[!] No episode returns data")
            return
        
        episodes = np.arange(1, len(returns) + 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Raw returns (light)
        ax.plot(episodes, returns, alpha=0.3, color='steelblue', linewidth=0.5, label='Raw Returns')
        
        # Moving average
        if len(returns) >= window:
            ma = np.convolve(returns, np.ones(window)/window, mode='valid')
            ma_episodes = np.arange(window, len(returns) + 1)
            ax.plot(ma_episodes, ma, linewidth=2.5, color='darkblue', 
                   label=f'{window}-Episode Moving Average')
        
        # Confidence intervals (std)
        if len(returns) >= window:
            std_window = []
            for i in range(window-1, len(returns)):
                std_window.append(np.std(returns[i-window+1:i+1]))
            std_window = np.array(std_window)
            ax.fill_between(ma_episodes, ma - std_window, ma + std_window, 
                          alpha=0.2, color='darkblue', label='±1 Std Dev')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title('Learning Curve: Episode Returns Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if len(returns) >= 20:
            early = np.mean(returns[:len(returns)//4])
            late = np.mean(returns[-len(returns)//4:])
            improvement = late - early
            ax.text(0.02, 0.98, 
                   f'Early (Q1): {early:.2f}\nLate (Q4): {late:.2f}\nImprovement: {improvement:+.2f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / '01_learning_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Learning curve -> {save_path.name}")
    
    def plot_episode_length_evolution(self):
        """
        Plot 2: Episode Length Evolution - Sample Efficiency
        
        Shows how episode duration changes over training.
        Longer episodes may indicate better exploration or policy quality.
        """
        lengths = np.array(self.metrics.get('episode_lengths', []))
        if len(lengths) == 0:
            print("[!] No episode length data")
            return
        
        episodes = np.arange(1, len(lengths) + 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, lengths, alpha=0.6, color='forestgreen', linewidth=1.5)
        
        window = min(50, max(10, len(lengths) // 10))
        if len(lengths) >= window:
            ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(lengths) + 1), ma, linewidth=2.5, 
                   color='darkgreen', label=f'{window}-Episode MA')
        
        mean_len = np.mean(lengths)
        ax.axhline(y=mean_len, color='gray', linestyle=':', alpha=0.5, 
                  label=f'Mean: {mean_len:.1f} turns')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length (turns)', fontsize=12)
        ax.set_title('Sample Efficiency: Episode Duration Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '02_episode_length.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Episode length -> {save_path.name}")
    
    # ========== 2. VALUE FUNCTION ANALYSIS ==========
    
    def plot_value_function_evolution(self):
        """
        Plot 3: Value Function Evolution
        
        Shows how the critic's value estimates change over training.
        Requires value estimates logged during training.
        """
        # Check if value data exists in training logs
        value_data = self.training_data.get('value_estimates', [])
        if not value_data:
            print("[!] No value function data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean value per episode
        episodes = np.arange(1, len(value_data) + 1)
        mean_values = [np.mean(v) if isinstance(v, list) else v for v in value_data]
        
        ax.plot(episodes, mean_values, alpha=0.6, color='purple', linewidth=2)
        
        window = min(20, max(5, len(mean_values) // 10))
        if len(mean_values) >= window:
            ma = np.convolve(mean_values, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(mean_values) + 1), ma, linewidth=2.5, 
                   color='darkviolet', label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Mean Value Estimate', fontsize=12)
        ax.set_title('Value Function Evolution: Critic Estimates Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '03_value_function.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Value function -> {save_path.name}")
    
    # ========== 3. POLICY GRADIENT METRICS ==========
    
    def plot_policy_loss(self):
        """
        Plot 4: Policy Loss Over Training
        
        Shows actor (policy) loss evolution.
        Decreasing loss indicates improving policy.
        """
        loss_data = self.training_data.get('policy_losses', [])
        if not loss_data:
            print("[!] No policy loss data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        updates = np.arange(1, len(loss_data) + 1)
        ax.plot(updates, loss_data, alpha=0.6, color='crimson', linewidth=1.5)
        
        window = min(50, max(10, len(loss_data) // 10))
        if len(loss_data) >= window:
            ma = np.convolve(loss_data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(loss_data) + 1), ma, linewidth=2.5, 
                   color='darkred', label=f'{window}-Update MA')
        
        ax.set_xlabel('Training Update', fontsize=12)
        ax.set_ylabel('Policy Loss', fontsize=12)
        ax.set_title('Policy Gradient Loss Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '04_policy_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Policy loss -> {save_path.name}")
    
    def plot_value_loss(self):
        """
        Plot 5: Value Loss Over Training
        
        Shows critic (value) loss evolution.
        Decreasing loss indicates better value estimation.
        """
        loss_data = self.training_data.get('value_losses', [])
        if not loss_data:
            print("[!] No value loss data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        updates = np.arange(1, len(loss_data) + 1)
        ax.plot(updates, loss_data, alpha=0.6, color='teal', linewidth=1.5)
        
        window = min(50, max(10, len(loss_data) // 10))
        if len(loss_data) >= window:
            ma = np.convolve(loss_data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(loss_data) + 1), ma, linewidth=2.5, 
                   color='darkcyan', label=f'{window}-Update MA')
        
        ax.set_xlabel('Training Update', fontsize=12)
        ax.set_ylabel('Value Loss', fontsize=12)
        ax.set_title('Value Function Loss Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '05_value_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Value loss -> {save_path.name}")
    
    def plot_policy_entropy(self):
        """
        Plot 6: Policy Entropy Over Training
        
        Shows exploration vs exploitation trade-off.
        High entropy = more exploration, low entropy = more exploitation.
        """
        entropy_data = self.training_data.get('entropies', [])
        if not entropy_data:
            print("[!] No entropy data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        updates = np.arange(1, len(entropy_data) + 1)
        ax.plot(updates, entropy_data, alpha=0.6, color='orange', linewidth=1.5)
        
        window = min(50, max(10, len(entropy_data) // 10))
        if len(entropy_data) >= window:
            ma = np.convolve(entropy_data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(entropy_data) + 1), ma, linewidth=2.5, 
                   color='darkorange', label=f'{window}-Update MA')
        
        ax.set_xlabel('Training Update', fontsize=12)
        ax.set_ylabel('Policy Entropy', fontsize=12)
        ax.set_title('Policy Entropy: Exploration vs Exploitation', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '06_policy_entropy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Policy entropy -> {save_path.name}")
    
    # ========== 4. ADVANTAGE ANALYSIS ==========
    
    def plot_advantage_distribution(self):
        """
        Plot 7: Advantage Distribution
        
        Histogram of advantage values.
        Helps understand value function quality and policy gradient signals.
        """
        advantage_data = self.training_data.get('advantages', [])
        if not advantage_data:
            print("[!] No advantage data available")
            return
        
        # Flatten if nested
        if isinstance(advantage_data[0], list):
            advantages = [a for sublist in advantage_data for a in sublist]
        else:
            advantages = advantage_data
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(advantages, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Advantage')
        ax.axvline(x=np.mean(advantages), color='green', linestyle='-', linewidth=2, 
                  label=f'Mean: {np.mean(advantages):.3f}')
        
        ax.set_xlabel('Advantage Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Advantage Distribution: Policy Gradient Signal Quality', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / '07_advantage_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Advantage distribution -> {save_path.name}")
    
    # ========== 5. HRL-SPECIFIC PLOTS ==========
    
    def plot_option_transition_matrix(self):
        """
        Plot 8: Option Transition Matrix (HRL)
        
        Heatmap showing transitions between high-level options.
        Reveals hierarchical policy structure and strategy patterns.
        """
        transitions = self.metrics.get('option_transitions', {})
        if not transitions:
            print("[!] No option transition data")
            return
        
        # Build transition matrix
        all_options = set()
        for from_opt in transitions:
            all_options.add(from_opt)
            for to_opt in transitions[from_opt]:
                all_options.add(to_opt)
        
        all_options = sorted(list(all_options))
        if len(all_options) == 0:
            print("[!] No options found")
            return
        
        matrix = np.zeros((len(all_options), len(all_options)))
        for i, from_opt in enumerate(all_options):
            for j, to_opt in enumerate(all_options):
                matrix[i, j] = transitions.get(from_opt, {}).get(to_opt, 0)
        
        # Normalize rows (probability of transitioning FROM each option)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / row_sums
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count matrix
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=all_options, yticklabels=all_options,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Option Transition Counts', fontweight='bold', pad=15)
        ax1.set_xlabel('To Option', fontsize=12)
        ax1.set_ylabel('From Option', fontsize=12)
        
        # Probability matrix
        sns.heatmap(matrix_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=all_options, yticklabels=all_options,
                   ax=ax2, cbar_kws={'label': 'Probability'})
        ax2.set_title('Option Transition Probabilities', fontweight='bold', pad=15)
        ax2.set_xlabel('To Option', fontsize=12)
        ax2.set_ylabel('From Option', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / '08_option_transitions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Option transitions -> {save_path.name}")
    
    def plot_option_duration_distribution(self):
        """
        Plot 9: Option Duration Distribution (HRL)
        
        Shows how long each option persists before termination.
        Longer durations indicate better temporal abstraction.
        """
        durations = self.metrics.get('option_durations', {})
        if not durations or all(len(v) == 0 for v in durations.values()):
            print("[!] No option duration data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        options = [k for k, v in durations.items() if len(v) > 0]
        duration_lists = [durations[k] for k in options]
        
        bp = ax.boxplot(duration_lists, labels=options, patch_artist=True, 
                       showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        colors = sns.color_palette("Set2", len(options))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Option Duration (turns)', fontsize=12)
        ax.set_xlabel('High-Level Option', fontsize=12)
        ax.set_title('Option Persistence: Temporal Abstraction Quality', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean annotations
        for i, (opt, durs) in enumerate(zip(options, duration_lists)):
            mean_dur = np.mean(durs)
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_dur:.1f}',
                   ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        save_path = self.output_dir / '09_option_durations.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Option durations -> {save_path.name}")
    
    # ========== 6. REWARD ANALYSIS ==========
    
    def plot_reward_distribution(self):
        """
        Plot 10: Reward Distribution
        
        Histogram of episode rewards.
        Shows reward variability and distribution shape.
        """
        returns = np.array(self.metrics.get('episode_returns', []))
        if len(returns) == 0:
            print("[!] No episode returns data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=np.mean(returns), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(returns):.2f}')
        ax.axvline(x=np.median(returns), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(returns):.2f}')
        
        ax.set_xlabel('Episode Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Reward Distribution: Performance Variability', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / '10_reward_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Reward distribution -> {save_path.name}")
    
    def plot_reward_decomposition(self):
        """
        Plot 11: Reward Component Decomposition
        
        Shows contribution of each reward component over time.
        """
        returns = np.array(self.metrics.get('episode_returns', []))
        if len(returns) == 0:
            print("[!] No episode returns data")
            return
        
        # Extract reward components if available
        components = {}
        for comp in ['engagement', 'novelty', 'responsiveness', 'transition', 'conclude']:
            comp_data = self.metrics.get(f'episode_{comp}_reward', [])
            if comp_data:
                components[comp] = np.array(comp_data)
        
        if not components:
            print("[!] No reward component data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(1, len(returns) + 1)
        colors = sns.color_palette("Set2", len(components))
        
        for (comp, values), color in zip(components.items(), colors):
            if len(values) == len(episodes):
                ax.plot(episodes, values, label=comp.capitalize(), color=color, linewidth=2)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward Component Value', fontsize=12)
        ax.set_title('Reward Decomposition: Component Contributions Over Training', 
                    fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '11_reward_decomposition.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Reward decomposition -> {save_path.name}")
    
    # ========== 7. TRAINING STABILITY ==========
    
    def plot_training_stability(self):
        """
        Plot 12: Training Stability Metrics
        
        Shows variance, standard deviation, and convergence indicators.
        """
        returns = np.array(self.metrics.get('episode_returns', []))
        if len(returns) == 0:
            print("[!] No episode returns data")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = np.arange(1, len(returns) + 1)
        window = min(50, max(10, len(returns) // 10))
        
        # Rolling statistics
        rolling_mean = []
        rolling_std = []
        
        for i in range(len(returns)):
            start = max(0, i - window + 1)
            window_data = returns[start:i+1]
            rolling_mean.append(np.mean(window_data))
            rolling_std.append(np.std(window_data))
        
        # Plot 1: Mean and std bands
        ax1.plot(episodes, rolling_mean, linewidth=2, color='darkblue', label='Rolling Mean')
        ax1.fill_between(episodes, 
                        np.array(rolling_mean) - np.array(rolling_std),
                        np.array(rolling_mean) + np.array(rolling_std),
                        alpha=0.3, color='steelblue', label='±1 Std Dev')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Return', fontsize=12)
        ax1.set_title('Training Stability: Rolling Mean and Variance', fontweight='bold', pad=15)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient of variation (std/mean)
        cv = np.array(rolling_std) / (np.array(rolling_mean) + 1e-8)
        ax2.plot(episodes, cv, linewidth=2, color='crimson')
        ax2.axhline(y=np.mean(cv), color='gray', linestyle='--', alpha=0.5,
                   label=f'Mean CV: {np.mean(cv):.3f}')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Coefficient of Variation (σ/μ)', fontsize=12)
        ax2.set_title('Training Stability: Coefficient of Variation', fontweight='bold', pad=15)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / '12_training_stability.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Training stability -> {save_path.name}")
    
    # ========== 8. POLICY ACTION DISTRIBUTION ==========
    
    def plot_action_distribution(self):
        """
        Plot 13: Action Distribution
        
        Shows frequency of each action.
        For hierarchical RL: shows option distribution
        For flat RL: shows flat action distribution
        """
        if self.is_flat_rl:
            # For flat RL, use flat_action_counts or reconstruct from option_counts
            flat_action_counts = self.metrics.get('flat_action_counts', {})
            
            # If not directly available, try to get from option_counts
            # (in flat RL, option_counts might contain flat action names)
            if not flat_action_counts:
                option_counts = self.metrics.get('option_counts', {})
                # Check if these look like flat action names (e.g., "Explain_ExplainNewFact")
                if option_counts:
                    # Check if any key contains underscore (flat action format)
                    has_flat_format = any('_' in str(k) for k in option_counts.keys())
                    if has_flat_format:
                        flat_action_counts = option_counts
            
            if not flat_action_counts:
                print("[!] No flat action usage data")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Flat action distribution
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
            save_path = self.output_dir / '13_action_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Flat action distribution -> {save_path.name}")
        else:
            # Hierarchical RL: show option distribution
            option_counts = self.metrics.get('option_counts', {})
            if not option_counts:
                print("[!] No option usage data")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Option distribution
            options = list(option_counts.keys())
            counts = list(option_counts.values())
            total = sum(counts)
            percentages = [c/total*100 for c in counts]
            
            colors = sns.color_palette("Set2", len(options))
            bars = ax1.bar(options, percentages, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_ylabel('Usage Percentage (%)', fontsize=12)
            ax1.set_xlabel('High-Level Option', fontsize=12)
            ax1.set_title('Option Usage Distribution', fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Pie chart
            ax2.pie(percentages, labels=options, autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Option Usage Proportion', fontweight='bold', pad=15)
            
            plt.tight_layout()
            save_path = self.output_dir / '13_action_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[+] Action distribution -> {save_path.name}")
    
    # ========== MAIN GENERATOR ==========
    
    def generate_all_plots(self):
        """Generate all RL/HRL evaluation plots"""
        print("\n" + "="*80)
        print("GENERATING RL/HRL EVALUATION PLOTS")
        print("="*80)
        print(f"Output directory: {self.output_dir}\n")
        
        # Learning curves
        print("[1] Learning Curves:")
        self.plot_learning_curve()
        self.plot_episode_length_evolution()
        
        # Value function
        print("\n[2] Value Function Analysis:")
        self.plot_value_function_evolution()
        
        # Policy gradients
        print("\n[3] Policy Gradient Metrics:")
        self.plot_policy_loss()
        self.plot_value_loss()
        self.plot_policy_entropy()
        
        # Advantage
        print("\n[4] Advantage Analysis:")
        self.plot_advantage_distribution()
        
        # HRL-specific
        print("\n[5] HRL-Specific Analysis:")
        self.plot_option_transition_matrix()
        self.plot_option_duration_distribution()
        
        # Reward analysis
        print("\n[6] Reward Analysis:")
        self.plot_reward_distribution()
        self.plot_reward_decomposition()
        
        # Training stability
        print("\n[7] Training Stability:")
        self.plot_training_stability()
        
        # Action distribution
        print("\n[8] Policy Analysis:")
        self.plot_action_distribution()
        
        print("\n" + "="*80)
        print(f"[OK] All plots saved to: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate RL/HRL evaluation plots')
    parser.add_argument('experiment_path', type=str, 
                       help='Path to experiment directory (e.g., training_logs/experiments/20251105/exp_003_20251105_161942)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: <experiment_path>/RL_plots)')
    
    args = parser.parse_args()
    
    generator = RLPlotGenerator(args.experiment_path, args.output)
    generator.generate_all_plots()


if __name__ == '__main__':
    main()
