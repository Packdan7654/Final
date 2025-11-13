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
        
        # RL-specific metrics (for RL plots)
        self.value_losses = []  # Critic loss per update
        self.policy_losses = []  # Actor loss per update
        self.entropies = []  # Policy entropy per update
        self.value_estimates = []  # Mean value estimate per episode
        self.advantages = []  # Advantage values per update
        self.termination_losses = []  # Termination function loss
        
        # Enhanced RL metrics tracking
        # Convergence metrics
        self.convergence_episode = None
        self.convergence_samples = None
        self.convergence_time_seconds = None
        self.convergence_window_mean = None
        self.convergence_window_std = None
        
        # Learning dynamics
        self.gradient_norms = []  # L2 norm of gradients per update
        self.parameter_update_norms = []  # L2 norm of parameter updates per update
        self.td_errors = []  # Temporal difference errors per update
        self.value_function_accuracy = None  # MSE between predicted and actual returns
        
        # Training efficiency
        self.samples_per_second = []
        self.updates_per_episode = []
        self.time_per_episode = []  # Wall-clock time per episode
        self.total_training_time_seconds = None
        self.time_to_target_return = None
        
        # Policy learning
        self.exploration_rate = []  # Action entropy normalized by max entropy
        self.exploitation_ratio = []  # Ratio of greedy vs exploratory actions
        self.option_learning_curves = defaultdict(list)  # Per-option return over time
        self.subaction_learning_curves = defaultdict(list)  # Per-subaction return over time
        
        # Value function learning
        self.value_function_variance = []  # Variance in value estimates
        self.value_bias = []  # Difference between predicted and actual returns
        self.value_convergence_episode = None
        
        # Stability metrics
        self.training_stability_score = None
        self.loss_spikes = []  # Episodes with unusually high losses
        self.gradient_explosions = []  # Episodes with gradient norm > threshold
        
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
        
        # Update RL-specific metrics
        if "mean_value" in episode_data:
            self.value_estimates.append(episode_data["mean_value"])
        
        # Update success rates
        if "transition_attempts" in episode_data:
            self.transition_attempts += episode_data["transition_attempts"]
            self.transition_successes += episode_data.get("transition_successes", 0)
        
        if "question_deflections" in episode_data:
            self.question_deflections += episode_data["question_deflections"]
            self.question_answers += episode_data.get("question_answers", 0)
            
        if "hallucinations" in episode_data:
            self.hallucination_counts.append(episode_data["hallucinations"])
    
    def update_training_stats(self, stats: Dict):
        """Update with training statistics from ActorCriticTrainer"""
        if "value_loss" in stats:
            self.value_losses.append(stats["value_loss"])
        if "policy_loss" in stats:
            self.policy_losses.append(stats["policy_loss"])
        if "entropy" in stats:
            self.entropies.append(stats["entropy"])
        if "termination_loss" in stats:
            self.termination_losses.append(stats["termination_loss"])
        if "mean_advantage" in stats:
            self.advantages.append(stats["mean_advantage"])
        
        # Enhanced RL metrics
        if "gradient_norm" in stats:
            self.gradient_norms.append(stats["gradient_norm"])
        if "update_norm" in stats:
            self.parameter_update_norms.append(stats["update_norm"])
        if "td_error" in stats:
            if isinstance(stats["td_error"], (list, np.ndarray)):
                self.td_errors.extend(stats["td_error"] if isinstance(stats["td_error"], list) else stats["td_error"].tolist())
            else:
                self.td_errors.append(stats["td_error"])
    
    def update_turn(self, turn_data: Dict):
        """Update with single turn data"""
        self.turn_rewards.append(turn_data.get("total_reward", 0.0))
        self.turn_dwells.append(turn_data.get("dwell", 0.0))
        
        # For flat RL, use flat_action_name if available, otherwise use option
        flat_action_name = turn_data.get("flat_action_name")
        if flat_action_name:
            # Flat RL: track flat actions
            action_key = flat_action_name
            if not hasattr(self, 'flat_action_counts'):
                from collections import defaultdict
                self.flat_action_counts = defaultdict(int)
            self.flat_action_counts[action_key] += 1
            # Also store in option_counts for backward compatibility
            self.option_counts[action_key] += 1
            self.turn_options.append(action_key)
            self.current_episode_options[action_key] += 1
        else:
            # Hierarchical RL: track options
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
            
            # Reward decomposition (total and average per episode)
            "reward_breakdown": {
                k: np.sum(v) for k, v in self.reward_components.items()
            },
            "reward_breakdown_avg": {
                k: np.mean(v) if len(v) > 0 else 0.0 for k, v in self.reward_components.items()
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
            "flat_action_counts": dict(getattr(self, 'flat_action_counts', {})),
            "option_durations": {k: list(v) for k, v in self.option_durations.items()},
            "option_transitions": {k: dict(v) for k, v in self.option_transitions.items()},
            # RL-specific metrics
            "value_losses": self.value_losses,
            "policy_losses": self.policy_losses,
            "entropies": self.entropies,
            "value_estimates": self.value_estimates,
            "advantages": self.advantages,
            "termination_losses": self.termination_losses,
            "summary": self.get_summary_statistics()
        }
        
        # Convert all numpy types to JSON-serializable Python types
        data = convert_to_json_serializable(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Metrics saved to {filepath}")
    
    def update_convergence_metrics(self, convergence_data: Dict):
        """Store convergence analysis results"""
        self.convergence_episode = convergence_data.get("episode")
        self.convergence_samples = convergence_data.get("samples")
        self.convergence_time_seconds = convergence_data.get("time_seconds")
        self.convergence_window_mean = convergence_data.get("window_mean")
        self.convergence_window_std = convergence_data.get("window_std")
    
    def update_learning_dynamics(self, gradient_norm: float = None, 
                                 update_norm: float = None, 
                                 td_error: float = None):
        """Store learning dynamics metrics"""
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        if update_norm is not None:
            self.parameter_update_norms.append(update_norm)
        if td_error is not None:
            if isinstance(td_error, (list, np.ndarray)):
                self.td_errors.extend(td_error if isinstance(td_error, list) else td_error.tolist())
            else:
                self.td_errors.append(td_error)
    
    def update_training_efficiency(self, samples_per_sec: float = None,
                                   time_per_ep: float = None):
        """Store training efficiency metrics"""
        if samples_per_sec is not None:
            self.samples_per_second.append(samples_per_sec)
        if time_per_ep is not None:
            self.time_per_episode.append(time_per_ep)
    
    def get_rl_metrics_summary(self) -> Dict:
        """Return comprehensive RL metrics dictionary"""
        return {
            "convergence": {
                "episode": self.convergence_episode,
                "samples": self.convergence_samples,
                "time_seconds": self.convergence_time_seconds,
                "window_mean": self.convergence_window_mean,
                "window_std": self.convergence_window_std
            },
            "learning_dynamics": {
                "gradient_norms": self.gradient_norms,
                "update_norms": self.parameter_update_norms,
                "td_errors": self.td_errors,
                "value_accuracy": self.value_function_accuracy
            },
            "training_efficiency": {
                "samples_per_second": self.samples_per_second,
                "updates_per_episode": self.updates_per_episode,
                "time_per_episode": self.time_per_episode,
                "total_time_seconds": self.total_training_time_seconds,
                "time_to_target_return": self.time_to_target_return
            },
            "policy_learning": {
                "entropy_over_time": self.entropies,
                "exploration_rate": self.exploration_rate,
                "exploitation_ratio": self.exploitation_ratio
            },
            "value_learning": {
                "value_estimates": self.value_estimates,
                "value_variance": self.value_function_variance,
                "value_bias": self.value_bias,
                "convergence_episode": self.value_convergence_episode
            },
            "stability": {
                "stability_score": self.training_stability_score,
                "loss_spikes": self.loss_spikes,
                "gradient_explosions": self.gradient_explosions
            }
        }