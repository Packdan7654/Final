"""
HRL Evaluation Visualization System
Creates thesis-ready plots organized by research questions and metrics
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class HRLEvaluationPlotter:
    """
    Creates structured evaluation plots for HRL museum dialogue agent
    Based on RQs from paper.tex:
    - RQ1: Long-horizon coherence (option persistence, strategy alignment)
    - RQ2: Engagement responsiveness (dwell-based switching, adaptation)
    - RQ3: Content quality (coverage, novelty, non-redundancy)
    """
    
    def __init__(self, experiment_dir):
        self.exp_dir = Path(experiment_dir)
        self.eval_dir = self.exp_dir / 'evaluation'
        
        # Create evaluation directory structure
        self.dirs = {
            'learning': self.eval_dir / '01_learning_dynamics',
            'hierarchy': self.eval_dir / '02_hierarchical_control',
            'engagement': self.eval_dir / '03_engagement_adaptation',
            'content': self.eval_dir / '04_content_quality',
            'reward': self.eval_dir / '05_reward_decomposition',
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[+] Evaluation directory structure created:")
        for name, path in self.dirs.items():
            print(f"   {name}: {path.name}")
    
    def load_data(self):
        """Load training data from experiment folder"""
        # Load metrics
        metrics_file = list((self.exp_dir / 'logs').glob('metrics_tracker_*.json'))
        if not metrics_file:
            raise FileError(f"No metrics found in {self.exp_dir / 'logs'}")
        
        with open(metrics_file[0], 'r') as f:
            self.metrics = json.load(f)
        
        # Load turn-by-turn data
        monitor_turns = list((self.exp_dir / 'logs').glob('monitor_*_turns_*.json'))
        if monitor_turns:
            with open(monitor_turns[0], 'r') as f:
                self.turn_data = json.load(f)
        else:
            self.turn_data = []
        
        # Load episode data
        monitor_episodes = list((self.exp_dir / 'logs').glob('monitor_*_episodes_*.json'))
        if monitor_episodes:
            with open(monitor_episodes[0], 'r') as f:
                self.episode_data = json.load(f)
        else:
            self.episode_data = []
        
        print(f"[OK] Loaded data:")
        print(f"   Episodes: {len(self.metrics.get('episode_returns', []))}")
        print(f"   Turns: {len(self.turn_data)}")
    
    # ========== 1. LEARNING DYNAMICS ==========
    
    def plot_learning_curve(self):
        """Core RL metric: Cumulative reward over episodes"""
        returns = np.array(self.metrics['episode_returns'])
        episodes = np.arange(1, len(returns) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Raw returns (light)
        ax.plot(episodes, returns, alpha=0.3, color='steelblue', linewidth=0.5)
        
        # Moving average (emphasized)
        if len(returns) >= 10:
            window = min(50, max(3, len(returns) // 10))
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(returns) + 1), moving_avg, 
                   linewidth=2.5, color='darkblue', label=f'{window}-episode MA')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Learning Curve: Agent Performance Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add performance stats
        if len(returns) >= 20:
            early_perf = np.mean(returns[:len(returns)//4])
            late_perf = np.mean(returns[-len(returns)//4:])
            improvement = late_perf - early_perf
            ax.text(0.02, 0.98, 
                   f'Early quarter: {early_perf:.2f}\nLate quarter: {late_perf:.2f}\nImprovement: {improvement:+.2f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.dirs['learning'] / 'learning_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Learning curve -> {save_path.name}")
    
    def plot_episode_length(self):
        """Sample efficiency: Turns per episode over training"""
        lengths = np.array(self.metrics['episode_lengths'])
        episodes = np.arange(1, len(lengths) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, lengths, alpha=0.6, color='forestgreen', linewidth=1.5)
        
        if len(lengths) >= 10:
            window = min(20, max(3, len(lengths) // 10))
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(lengths) + 1), moving_avg,
                   linewidth=2.5, color='darkgreen', label=f'{window}-episode MA')
        
        ax.axhline(y=np.mean(lengths), color='gray', linestyle=':', alpha=0.5, label=f'Mean: {np.mean(lengths):.1f} turns')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length (turns)')
        ax.set_title('Sample Efficiency: Episode Duration Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['learning'] / 'episode_length.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Episode length -> {save_path.name}")
    
    # ========== 2. HIERARCHICAL CONTROL (RQ1) ==========
    
    def plot_option_usage_distribution(self):
        """RQ1: Strategy evolution - how does option usage change over training?"""
        episode_option_usage = self.metrics.get('episode_option_usage', [])
        
        if not episode_option_usage:
            print("   [!] No per-episode option usage data")
            return
        
        # Get all unique options
        all_options = set()
        for ep_usage in episode_option_usage:
            all_options.update(ep_usage.keys())
        all_options = sorted(list(all_options))
        
        if not all_options:
            print("   [!] No option data found")
            return
        
        # Calculate percentages per episode
        episodes = np.arange(1, len(episode_option_usage) + 1)
        option_percentages = {opt: [] for opt in all_options}
        
        for ep_usage in episode_option_usage:
            total = sum(ep_usage.values())
            if total == 0:
                for opt in all_options:
                    option_percentages[opt].append(0)
            else:
                for opt in all_options:
                    pct = (ep_usage.get(opt, 0) / total) * 100
                    option_percentages[opt].append(pct)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("Set2", len(all_options))
        for i, opt in enumerate(all_options):
            ax.plot(episodes, option_percentages[opt], 
                   label=opt, color=colors[i], linewidth=2.5, marker='o', markersize=4)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Usage Frequency (%)', fontsize=12)
        ax.set_title('RQ1: Option Usage Evolution Over Training', fontweight='bold', pad=15, fontsize=14)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        
        # Add reference line at 25% (equal distribution for 4 options)
        if len(all_options) > 0:
            equal_pct = 100 / len(all_options)
            ax.axhline(y=equal_pct, color='gray', linestyle='--', alpha=0.5, 
                      label=f'Equal distribution ({equal_pct:.0f}%)')
        
        plt.tight_layout()
        save_path = self.dirs['hierarchy'] / 'option_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Option distribution -> {save_path.name}")
    
    def plot_option_persistence(self):
        """RQ1: How long does each option persist? (Multi-turn coherence)"""
        option_durations = self.metrics.get('option_durations', {})
        
        if not option_durations or all(len(v) == 0 for v in option_durations.values()):
            print("   [!] No option duration data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        options = [k for k, v in option_durations.items() if len(v) > 0]
        durations = [v for k, v in option_durations.items() if len(v) > 0]
        
        bp = ax.boxplot(durations, labels=options, patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        
        colors = sns.color_palette("Set2", len(options))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Option Duration (turns)')
        ax.set_xlabel('High-Level Option')
        ax.set_title('RQ1: Option Persistence (Multi-Turn Coherence)', fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean values as text
        for i, (opt, durs) in enumerate(zip(options, durations)):
            mean_dur = np.mean(durs)
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_dur:.1f}',
                   ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        save_path = self.dirs['hierarchy'] / 'option_persistence.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Option persistence -> {save_path.name}")
    
    def plot_option_evolution(self):
        """RQ1: How does option usage evolve during training?"""
        if not self.turn_data:
            print("   [!] No turn data for option evolution")
            return
        
        # Bin episodes
        n_episodes = len(self.metrics['episode_returns'])
        n_bins = min(10, max(3, n_episodes // 20))
        
        # Count options per episode bin
        episode_bins = np.linspace(0, n_episodes, n_bins + 1, dtype=int)
        option_counts_per_bin = []
        
        current_ep = 0
        for i, turn in enumerate(self.turn_data):
            if turn.get('done', False):
                current_ep += 1
        
        # Reconstruct option counts per bin
        option_names = set()
        current_ep = 0
        bin_idx = 0
        current_bin_counts = defaultdict(int)
        
        for turn in self.turn_data:
            option = turn.get('info', {}).get('option', 'Unknown')
            option_names.add(option)
            current_bin_counts[option] += 1
            
            if turn.get('done', False):
                current_ep += 1
                if bin_idx < len(episode_bins) - 1 and current_ep >= episode_bins[bin_idx + 1]:
                    option_counts_per_bin.append(dict(current_bin_counts))
                    current_bin_counts = defaultdict(int)
                    bin_idx += 1
        
        if current_bin_counts:
            option_counts_per_bin.append(dict(current_bin_counts))
        
        if len(option_counts_per_bin) < 2:
            print("   [!] Insufficient data for option evolution")
            return
        
        # Plot evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        option_names = sorted(option_names - {'Unknown'})
        x = np.arange(len(option_counts_per_bin))
        
        for opt in option_names:
            percentages = []
            for bin_counts in option_counts_per_bin:
                total = sum(bin_counts.values())
                percentages.append(bin_counts.get(opt, 0) / max(total, 1) * 100)
            ax.plot(x, percentages, marker='o', linewidth=2, label=opt)
        
        ax.set_xlabel('Training Progress (Episode Bins)')
        ax.set_ylabel('Option Usage (%)')
        ax.set_title('RQ1: Option Usage Evolution During Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{episode_bins[i]}-{episode_bins[i+1]}' for i in range(len(episode_bins)-1)])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_path = self.dirs['hierarchy'] / 'option_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Option evolution -> {save_path.name}")
    
    # ========== 3. ENGAGEMENT ADAPTATION (RQ2) ==========
    
    def plot_dwell_over_training(self):
        """RQ2: Does engagement (dwell) improve over training?"""
        if not self.turn_data:
            print("   [!] No turn data for dwell analysis")
            return
        
        # Extract dwell per episode
        episode_dwell = []
        current_ep_dwell = []
        
        for turn in self.turn_data:
            dwell = turn.get('observation', {}).get('dwell', 0.0)
            current_ep_dwell.append(dwell)
            
            if turn.get('done', False):
                if current_ep_dwell:
                    episode_dwell.append(np.mean(current_ep_dwell))
                current_ep_dwell = []
        
        if not episode_dwell:
            print("   [!] No dwell data")
            return
        
        episodes = np.arange(1, len(episode_dwell) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, episode_dwell, alpha=0.4, color='purple', linewidth=0.8)
        
        if len(episode_dwell) >= 10:
            window = min(20, max(3, len(episode_dwell) // 10))
            moving_avg = np.convolve(episode_dwell, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(episode_dwell) + 1), moving_avg,
                   linewidth=2.5, color='darkviolet', label=f'{window}-episode MA')
        
        ax.axhline(y=np.mean(episode_dwell), color='gray', linestyle=':', alpha=0.5, 
                  label=f'Mean: {np.mean(episode_dwell):.3f}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Dwell Time per Episode')
        ax.set_title('RQ2: Engagement (Dwell) Over Training', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['engagement'] / 'dwell_over_training.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Dwell over training -> {save_path.name}")
    
    # ========== 4. CONTENT QUALITY (RQ3) ==========
    
    def plot_exhibit_coverage(self):
        """RQ3: How many exhibits does agent cover per episode?"""
        coverage = np.array(self.metrics.get('episode_coverage', []))
        
        if len(coverage) == 0:
            print("   [!] No coverage data")
            return
        
        episodes = np.arange(1, len(coverage) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, coverage, alpha=0.6, color='teal', linewidth=1.5)
        
        if len(coverage) >= 10:
            window = min(20, max(3, len(coverage) // 10))
            moving_avg = np.convolve(coverage, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(coverage) + 1), moving_avg,
                   linewidth=2.5, color='darkcyan', label=f'{window}-episode MA')
        
        ax.axhline(y=np.mean(coverage), color='gray', linestyle=':', alpha=0.5,
                  label=f'Mean: {np.mean(coverage):.1%}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exhibit Coverage (%)')
        ax.set_title('RQ3: Museum Coverage - Exploring Diverse Content', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        save_path = self.dirs['content'] / 'exhibit_coverage.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Exhibit coverage -> {save_path.name}")
    
    def plot_facts_presented(self):
        """RQ3: Novelty - how many facts presented per episode?"""
        facts = np.array(self.metrics.get('episode_facts', []))
        
        if len(facts) == 0:
            print("   [!] No facts data")
            return
        
        episodes = np.arange(1, len(facts) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, facts, alpha=0.6, color='orange', linewidth=1.5)
        
        if len(facts) >= 10:
            window = min(20, max(3, len(facts) // 10))
            moving_avg = np.convolve(facts, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(facts) + 1), moving_avg,
                   linewidth=2.5, color='darkorange', label=f'{window}-episode MA')
        
        ax.axhline(y=np.mean(facts), color='gray', linestyle=':', alpha=0.5,
                  label=f'Mean: {np.mean(facts):.1f} facts')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('New Facts Presented')
        ax.set_title('RQ3: Content Novelty - Facts Presented Per Episode', fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['content'] / 'facts_presented.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Facts presented -> {save_path.name}")
    
    def plot_exhibit_coverage_heatmap(self):
        """RQ3: Heatmap showing coverage and facts per episode"""
        coverage = np.array(self.metrics.get('episode_coverage', []))
        facts = np.array(self.metrics.get('episode_facts', []))
        
        if len(coverage) == 0 or len(facts) == 0:
            print("   [!] No coverage/facts data for heatmap")
            return
        
        # Create a 2D array for visualization
        episodes = np.arange(1, len(coverage) + 1)
        
        # Combine metrics into a heatmap-style visualization
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Create scatter plot with size = facts, color = coverage
        scatter = ax.scatter(episodes, coverage * 100, 
                           s=facts * 20,  # Size based on facts
                           c=coverage * 100,  # Color based on coverage
                           cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add moving average line for coverage
        if len(coverage) >= 10:
            window = min(20, max(3, len(coverage) // 10))
            moving_avg = np.convolve(coverage, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(coverage) + 1), moving_avg * 100,
                   color='red', linewidth=2, label=f'Coverage Trend ({window}-ep MA)', alpha=0.7)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Coverage (%)', fontsize=11)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Exhibit Coverage (%)', fontsize=12)
        ax.set_title('RQ3: Coverage & Facts Evolution (bubble size = facts presented)', 
                    fontweight='bold', pad=15, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        save_path = self.dirs['content'] / 'coverage_facts_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Coverage-facts heatmap -> {save_path.name}")
    
    # ========== 5. REWARD DECOMPOSITION ==========
    
    def plot_reward_components(self):
        """Analyze reward signal composition"""
        if not self.turn_data:
            print("   [!] No turn data for reward decomposition")
            return
        
        # Extract reward components
        components = {
            'Engagement': [],
            'Novelty': [],
            'Responsiveness': [],
            'Other': []
        }
        
        for turn in self.turn_data:
            info = turn.get('info', {})
            components['Engagement'].append(info.get('reward_engagement', 0))
            components['Novelty'].append(info.get('reward_novelty', 0))
            components['Responsiveness'].append(info.get('reward_responsiveness', 0))
            other = (info.get('reward_conclude', 0) + 
                    info.get('reward_transition_penalty', 0))
            components['Other'].append(other)
        
        # Calculate means
        means = {k: np.mean(v) for k, v in components.items() if v}
        
        if not means:
            print("   [!] No reward component data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of average contribution
        positive_means = {k: v for k, v in means.items() if v > 0.001}
        if positive_means:
            colors = sns.color_palette("Set2", len(positive_means))
            ax1.pie(positive_means.values(), labels=positive_means.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Average Reward Composition', fontweight='bold', pad=15)
        
        # Bar chart of mean values
        comp_names = list(means.keys())
        comp_values = list(means.values())
        colors_bar = ['green' if v >= 0 else 'red' for v in comp_values]
        
        bars = ax2.barh(comp_names, comp_values, color=colors_bar, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Mean Reward Value')
        ax2.set_title('Reward Component Breakdown', fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, comp_values):
            x_pos = val + (0.01 if val >= 0 else -0.01)
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}',
                    va='center', ha='left' if val >= 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        save_path = self.dirs['reward'] / 'reward_decomposition.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Reward decomposition -> {save_path.name}")
    
    def plot_reward_over_training(self):
        """How do different reward components evolve over training?"""
        # Use episode-level reward data from metrics
        returns = np.array(self.metrics.get('episode_returns', []))
        lengths = np.array(self.metrics.get('episode_lengths', []))
        coverage = np.array(self.metrics.get('episode_coverage', []))
        facts = np.array(self.metrics.get('episode_facts', []))
        
        if len(returns) == 0:
            print("   [!] No episode reward data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Returns and episode length correlation
        episodes = np.arange(1, len(returns) + 1)
        
        # Plot returns
        color = 'tab:blue'
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Cumulative Return', color=color, fontsize=12)
        ax1.plot(episodes, returns, color=color, alpha=0.4, linewidth=0.8)
        if len(returns) >= 10:
            window = min(20, max(3, len(returns) // 10))
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(returns) + 1), moving_avg,
                   color=color, linewidth=2.5, label=f'Return ({window}-ep MA)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Plot episode length on secondary axis
        ax1_twin = ax1.twinx()
        color = 'tab:orange'
        ax1_twin.set_ylabel('Episode Length (turns)', color=color, fontsize=12)
        ax1_twin.plot(episodes, lengths, color=color, alpha=0.4, linewidth=0.8)
        if len(lengths) >= 10:
            window = min(20, max(3, len(lengths) // 10))
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax1_twin.plot(range(window, len(lengths) + 1), moving_avg,
                        color=color, linewidth=2.5, label=f'Length ({window}-ep MA)')
        ax1_twin.tick_params(axis='y', labelcolor=color)
        
        ax1.set_title('Return vs Episode Length Over Training', fontweight='bold', pad=15)
        
        # Plot 2: Content metrics (coverage and facts)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Exhibit Coverage (%)', color='tab:green', fontsize=12)
        if len(coverage) > 0:
            ax2.plot(episodes, coverage * 100, color='tab:green', alpha=0.4, linewidth=0.8)
            if len(coverage) >= 10:
                window = min(20, max(3, len(coverage) // 10))
                moving_avg = np.convolve(coverage, np.ones(window)/window, mode='valid')
                ax2.plot(range(window, len(coverage) + 1), moving_avg * 100,
                       color='tab:green', linewidth=2.5, label=f'Coverage ({window}-ep MA)')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.grid(True, alpha=0.3)
        
        # Plot facts on secondary axis
        ax2_twin = ax2.twinx()
        color = 'tab:purple'
        ax2_twin.set_ylabel('Facts Presented', color=color, fontsize=12)
        if len(facts) > 0:
            ax2_twin.plot(episodes, facts, color=color, alpha=0.4, linewidth=0.8)
            if len(facts) >= 10:
                window = min(20, max(3, len(facts) // 10))
                moving_avg = np.convolve(facts, np.ones(window)/window, mode='valid')
                ax2_twin.plot(range(window, len(facts) + 1), moving_avg,
                           color=color, linewidth=2.5, label=f'Facts ({window}-ep MA)')
        ax2_twin.tick_params(axis='y', labelcolor=color)
        
        ax2.set_title('Content Quality Metrics Over Training', fontweight='bold', pad=15)
        
        plt.tight_layout()
        save_path = self.dirs['reward'] / 'reward_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [+] Reward evolution -> {save_path.name}")
    
    def generate_all_plots(self):
        """Generate all evaluation plots"""
        print("\n" + "="*80)
        print("GENERATING THESIS EVALUATION PLOTS")
        print("="*80)
        
        print("\n[1] Learning Dynamics:")
        self.plot_learning_curve()
        self.plot_episode_length()
        
        print("\n[2] Hierarchical Control (RQ1):")
        self.plot_option_usage_distribution()
        self.plot_option_persistence()
        self.plot_option_evolution()
        
        print("\n[3] Engagement Adaptation (RQ2):")
        self.plot_dwell_over_training()
        
        print("\n[4] Content Quality (RQ3):")
        self.plot_exhibit_coverage()
        self.plot_facts_presented()
        self.plot_exhibit_coverage_heatmap()
        
        print("\n[5] Reward Decomposition:")
        self.plot_reward_components()
        self.plot_reward_over_training()
        
        print("\n" + "="*80)
        print(f"[OK] ALL PLOTS SAVED TO: {self.eval_dir}")
        print("="*80)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate text summary of key findings"""
        summary_path = self.eval_dir / 'EVALUATION_SUMMARY.txt'
        
        returns = np.array(self.metrics['episode_returns'])
        lengths = np.array(self.metrics['episode_lengths'])
        coverage = np.array(self.metrics.get('episode_coverage', []))
        facts = np.array(self.metrics.get('episode_facts', []))
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HRL TRAINING EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Experiment: {self.exp_dir.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("LEARNING PERFORMANCE:\n")
            f.write(f"  Total Episodes: {len(returns)}\n")
            f.write(f"  Mean Return: {np.mean(returns):.3f} ± {np.std(returns):.3f}\n")
            f.write(f"  Final 25% Mean: {np.mean(returns[-len(returns)//4:]):.3f}\n")
            if len(returns) >= 8:
                improvement = np.mean(returns[-len(returns)//4:]) - np.mean(returns[:len(returns)//4])
                f.write(f"  Improvement (early→late): {improvement:+.3f}\n")
            f.write(f"  Mean Episode Length: {np.mean(lengths):.1f} turns\n\n")
            
            if len(coverage) > 0:
                f.write("CONTENT METRICS (RQ3):\n")
                f.write(f"  Mean Exhibit Coverage: {np.mean(coverage):.1%}\n")
                f.write(f"  Mean Facts/Episode: {np.mean(facts):.1f}\n\n")
            
            option_counts = self.metrics.get('option_counts', {})
            if option_counts:
                total = sum(option_counts.values())
                f.write("OPTION USAGE (RQ1):\n")
                for opt, count in sorted(option_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  {opt}: {count/total*100:.1f}% ({count} times)\n")
                f.write(f"\n  Option Diversity: {len(option_counts)} unique options\n\n")
            
            f.write("="*80 + "\n")
            f.write("See subdirectories for detailed visualizations organized by RQ.\n")
            f.write("="*80 + "\n")
        
        print(f"\n[+] Summary saved: {summary_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate HRL evaluation plots')
    parser.add_argument('exp_number', type=int, help='Experiment number (e.g., 7 for exp_007)')
    args = parser.parse_args()
    
    # Find experiment directory
    exp_dir = Path('training_logs/experiments')
    exp_folders = sorted(exp_dir.glob(f'exp_{args.exp_number:03d}_*'))
    
    if not exp_folders:
        print(f"[ERROR] No experiment found for number {args.exp_number}")
        return
    
    target_exp = exp_folders[0]
    print(f"[*] Analyzing: {target_exp.name}\n")
    
    plotter = HRLEvaluationPlotter(target_exp)
    plotter.load_data()
    plotter.generate_all_plots()


if __name__ == '__main__':
    main()

