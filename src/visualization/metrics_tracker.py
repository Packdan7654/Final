# src/visualization/metrics_tracker.py

"""
Comprehensive Metrics Tracking for HRL Training

Tracks and computes:
- Episode-level metrics (returns, lengths, coverage)
- Turn-level metrics (rewards, actions, dwell)
- Option statistics (duration, success rates)
- Learning curves (moving averages, variance)
- Transition analysis
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from pathlib import Path


class MetricsTracker:
    """
    Tracks comprehensive metrics for HRL training analysis.
    
    Key Metrics:
    1. Episode Returns: cumulative, mean, std, min, max
    2. Episode Length: mean turns per episode
    3. Coverage: exhibits covered, facts mentioned
    4. Dwell: mean, median, distribution
    5. Option Usage: counts, durations, transitions
    6. Reward Components: breakdown by source
    7. Success Rates: transitions, question answering
    """
    
    def __init__(self):
        # Episode-level tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_coverage = []  # % of exhibits covered
        self.episode_facts = []  # total facts mentioned
        self.episode_option_usage = []  # Per-episode option counts: [{option: count}, ...]
        
        # Turn-level tracking
        self.turn_rewards = []
        self.turn_dwells = []
        self.turn_options = []
        
        # Reward decomposition
        self.reward_components = defaultdict(list)
        
        # Option statistics
        self.option_counts = defaultdict(int)
        self.option_durations = defaultdict(list)  # turns per option instance
        self.option_transitions = defaultdict(lambda: defaultdict(int))  # from -> to counts
        
        # Current episode option tracking (reset each episode)
        self.current_episode_options = defaultdict(int)
        
        # Success rates
        self.transition_attempts = 0
        self.transition_successes = 0
        self.question_deflections = 0
        self.question_answers = 0
        
        # Hallucination tracking
        self.hallucination_counts = []
        
    def update_episode(self, episode_data: Dict):
        """Update with complete episode data"""
        self.episode_returns.append(episode_data.get("cumulative_reward", 0.0))
        self.episode_lengths.append(episode_data.get("turns", 0))
        self.episode_coverage.append(episode_data.get("coverage_ratio", 0.0))
        self.episode_facts.append(episode_data.get("total_facts", 0))
        
        # Save current episode option usage and reset for next episode
        self.episode_option_usage.append(dict(self.current_episode_options))
        self.current_episode_options = defaultdict(int)
        
        # Update reward components
        for component in ["engagement", "novelty", "responsiveness", "transition", "conclude"]:
            value = episode_data.get(f"reward_{component}", 0.0)
            self.reward_components[component].append(value)
        
        # Update success rates
        if "transition_attempts" in episode_data:
            self.transition_attempts += episode_data["transition_attempts"]
            self.transition_successes += episode_data.get("transition_successes", 0)
        
        if "question_deflections" in episode_data:
            self.question_deflections += episode_data["question_deflections"]
            self.question_answers += episode_data.get("question_answers", 0)
            
        if "hallucinations" in episode_data:
            self.hallucination_counts.append(episode_data["hallucinations"])
    
    def update_turn(self, turn_data: Dict):
        """Update with single turn data"""
        self.turn_rewards.append(turn_data.get("total_reward", 0.0))
        self.turn_dwells.append(turn_data.get("dwell", 0.0))
        
        option = turn_data.get("option", "Unknown")
        self.turn_options.append(option)
        self.option_counts[option] += 1
        self.current_episode_options[option] += 1
        
    def update_option_transition(self, from_option: str, to_option: str, duration: int):
        """Track option-to-option transitions and durations"""
        self.option_transitions[from_option][to_option] += 1
        self.option_durations[from_option].append(duration)
    
    def get_summary_statistics(self, window: int = 100) -> Dict:
        """Get comprehensive summary statistics"""
        n_episodes = len(self.episode_returns)
        
        if n_episodes == 0:
            return {}
        
        # Recent window for trend analysis
        recent_returns = self.episode_returns[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_coverage = self.episode_coverage[-window:]
        
        summary = {
            # Overall statistics
            "total_episodes": n_episodes,
            "total_turns": sum(self.episode_lengths),
            
            # Returns
            "mean_return": np.mean(self.episode_returns),
            "std_return": np.std(self.episode_returns),
            "recent_mean_return": np.mean(recent_returns),
            "recent_std_return": np.std(recent_returns),
            
            # Episode length
            "mean_length": np.mean(self.episode_lengths),
            "recent_mean_length": np.mean(recent_lengths),
            
            # Coverage
            "mean_coverage": np.mean(self.episode_coverage),
            "recent_mean_coverage": np.mean(recent_coverage),
            "mean_facts_per_episode": np.mean(self.episode_facts),
            
            # Dwell
            "mean_dwell": np.mean(self.turn_dwells) if self.turn_dwells else 0.0,
            "median_dwell": np.median(self.turn_dwells) if self.turn_dwells else 0.0,
            
            # Option usage (proportions)
            "option_usage": self._compute_option_proportions(),
            
            # Option durations
            "option_mean_durations": self._compute_mean_durations(),
            
            # Success rates
            "transition_success_rate": (
                self.transition_successes / max(self.transition_attempts, 1)
            ),
            "question_answer_rate": (
                self.question_answers / max(self.question_answers + self.question_deflections, 1)
            ),
            
            # Hallucinations
            "mean_hallucinations_per_episode": (
                np.mean(self.hallucination_counts) if self.hallucination_counts else 0.0
            ),
            
            # Reward decomposition
            "reward_breakdown": {
                k: np.sum(v) for k, v in self.reward_components.items()
            }
        }
        
        return summary
    
    def _compute_option_proportions(self) -> Dict[str, float]:
        """Compute proportion of turns each option was used"""
        total = sum(self.option_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.option_counts.items()}
    
    def _compute_mean_durations(self) -> Dict[str, float]:
        """Compute mean duration (turns) for each option"""
        return {
            k: np.mean(v) if v else 0.0 
            for k, v in self.option_durations.items()
        }
    
    def get_learning_curve(self, window: int = 50) -> Tuple[List[float], List[float]]:
        """Get smoothed learning curve (returns over episodes)"""
        if len(self.episode_returns) < window:
            return self.episode_returns, [0] * len(self.episode_returns)
        
        smoothed = []
        stds = []
        
        for i in range(len(self.episode_returns)):
            start = max(0, i - window + 1)
            end = i + 1
            window_data = self.episode_returns[start:end]
            smoothed.append(np.mean(window_data))
            stds.append(np.std(window_data))
        
        return smoothed, stds
    
    def save_to_json(self, filepath: str):
        """Save all metrics to JSON"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        data = {
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "episode_coverage": self.episode_coverage,
            "episode_facts": self.episode_facts,
            "episode_option_usage": self.episode_option_usage,
            "option_counts": dict(self.option_counts),
            "option_durations": {k: list(v) for k, v in self.option_durations.items()},
            "summary": self.get_summary_statistics()
        }
        
        # Convert all numpy types to JSON-serializable Python types
        data = convert_to_json_serializable(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Metrics saved to {filepath}")