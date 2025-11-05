"""
Parameterization Analysis Tool for RL Reward Weight Tuning

This module provides analysis tools to help tune reward weights and other RL parameters
by analyzing training data and providing insights on:
- Reward component contributions
- Action distribution patterns
- Behavioral correlations
- Parameter sensitivity
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ParameterizationAnalyzer:
    """Analyzes training data to help tune RL parameters"""
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "parameterization_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.detailed_logs_dir = self.experiment_dir / "detailed_logs"
        self.metrics_dir = self.experiment_dir / "logs"
        
    def analyze_reward_components(self, episodes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze reward component breakdowns to understand what's driving rewards.
        
        Args:
            episodes: List of episode numbers to analyze (None = all episodes)
            
        Returns:
            Analysis results dictionary
        """
        reward_components = {
            "engagement": [],
            "novelty": [],
            "responsiveness": [],
            "conclude": [],
            "transition_insufficiency": []
        }
        
        total_rewards = []
        episode_rewards = []
        
        # Load episode data
        if not self.detailed_logs_dir.exists():
            return {"error": "Detailed logs not found"}
        
        episode_dirs = sorted(self.detailed_logs_dir.glob("episode_*"))
        if episodes:
            episode_dirs = [d for d in episode_dirs if int(d.name.split("_")[1]) in episodes]
        
        for ep_dir in episode_dirs:
            log_file = ep_dir / "episode_log.json"
            if not log_file.exists():
                continue
                
            with open(log_file, 'r', encoding='utf-8') as f:
                ep_data = json.load(f)
            
            ep_reward = 0.0
            for turn in ep_data.get("turns", []):
                reward = turn.get("reward", {})
                reward_components["engagement"].append(reward.get("engagement", 0.0))
                reward_components["novelty"].append(reward.get("novelty", 0.0))
                reward_components["responsiveness"].append(reward.get("responsiveness", 0.0))
                reward_components["conclude"].append(reward.get("conclude", 0.0))
                reward_components["transition_insufficiency"].append(reward.get("transition_insufficiency", 0.0))
                
                total_rewards.append(reward.get("total", 0.0))
                ep_reward += reward.get("total", 0.0)
            
            episode_rewards.append(ep_reward)
        
        # Calculate statistics
        stats = {}
        for component, values in reward_components.items():
            if values:
                stats[component] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "sum": np.sum(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "contribution_pct": (np.sum(values) / np.sum(total_rewards) * 100) if np.sum(total_rewards) != 0 else 0.0
                }
        
        stats["total_reward"] = {
            "mean": np.mean(total_rewards) if total_rewards else 0.0,
            "std": np.std(total_rewards) if total_rewards else 0.0,
            "sum": np.sum(total_rewards) if total_rewards else 0.0
        }
        
        stats["episode_reward"] = {
            "mean": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std": np.std(episode_rewards) if episode_rewards else 0.0,
            "min": np.min(episode_rewards) if episode_rewards else 0.0,
            "max": np.max(episode_rewards) if episode_rewards else 0.0
        }
        
        return {
            "reward_component_stats": stats,
            "recommendations": self._generate_reward_recommendations(stats)
        }
    
    def analyze_action_distributions(self, episodes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze action distribution patterns.
        
        Returns:
            Action distribution analysis
        """
        option_counts = defaultdict(int)
        subaction_counts = defaultdict(int)
        option_subaction_pairs = defaultdict(int)
        
        # Load episode data
        if not self.detailed_logs_dir.exists():
            return {"error": "Detailed logs not found"}
        
        episode_dirs = sorted(self.detailed_logs_dir.glob("episode_*"))
        if episodes:
            episode_dirs = [d for d in episode_dirs if int(d.name.split("_")[1]) in episodes]
        
        for ep_dir in episode_dirs:
            log_file = ep_dir / "episode_log.json"
            if not log_file.exists():
                continue
                
            with open(log_file, 'r', encoding='utf-8') as f:
                ep_data = json.load(f)
            
            for turn in ep_data.get("turns", []):
                action = turn.get("action", {})
                option = action.get("option", "Unknown")
                subaction = action.get("subaction", "Unknown")
                
                option_counts[option] += 1
                subaction_counts[subaction] += 1
                option_subaction_pairs[f"{option}/{subaction}"] += 1
        
        total_actions = sum(option_counts.values())
        
        return {
            "option_distribution": {k: {"count": v, "percentage": (v / total_actions * 100) if total_actions > 0 else 0.0} 
                                   for k, v in sorted(option_counts.items(), key=lambda x: -x[1])},
            "subaction_distribution": {k: {"count": v, "percentage": (v / total_actions * 100) if total_actions > 0 else 0.0}
                                      for k, v in sorted(subaction_counts.items(), key=lambda x: -x[1])},
            "option_subaction_pairs": dict(sorted(option_subaction_pairs.items(), key=lambda x: -x[1])),
            "recommendations": self._generate_action_recommendations(option_counts, subaction_counts, total_actions)
        }
    
    def analyze_novelty_patterns(self, episodes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze how well the agent is introducing new facts.
        
        Returns:
            Novelty analysis
        """
        explain_new_fact_counts = []
        total_facts_shared = []
        facts_per_explain = []
        
        # Load episode data
        if not self.detailed_logs_dir.exists():
            return {"error": "Detailed logs not found"}
        
        episode_dirs = sorted(self.detailed_logs_dir.glob("episode_*"))
        if episodes:
            episode_dirs = [d for d in episode_dirs if int(d.name.split("_")[1]) in episodes]
        
        for ep_dir in episode_dirs:
            log_file = ep_dir / "episode_log.json"
            if not log_file.exists():
                continue
                
            with open(log_file, 'r', encoding='utf-8') as f:
                ep_data = json.load(f)
            
            for turn in ep_data.get("turns", []):
                action = turn.get("action", {})
                facts = turn.get("facts", {})
                new_facts = len(facts.get("new_fact_ids", []))
                
                total_facts_shared.append(new_facts)
                
                if action.get("subaction") == "ExplainNewFact":
                    explain_new_fact_counts.append(new_facts)
                    facts_per_explain.append(new_facts)
        
        return {
            "explain_new_fact_usage": {
                "count": len(explain_new_fact_counts),
                "avg_facts_per_use": np.mean(explain_new_fact_counts) if explain_new_fact_counts else 0.0,
                "total_facts_introduced": sum(explain_new_fact_counts)
            },
            "overall_fact_sharing": {
                "avg_facts_per_turn": np.mean(total_facts_shared) if total_facts_shared else 0.0,
                "total_facts_shared": sum(total_facts_shared)
            },
            "recommendations": self._generate_novelty_recommendations(explain_new_fact_counts, total_facts_shared)
        }
    
    def _generate_reward_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on reward component analysis"""
        recommendations = []
        
        comp_stats = stats.get("reward_component_stats", {})
        
        # Check if novelty is too low
        novelty_contrib = comp_stats.get("novelty", {}).get("contribution_pct", 0.0)
        if novelty_contrib < 20.0:
            recommendations.append(
                f"⚠️  Novelty reward contributes only {novelty_contrib:.1f}% of total reward. "
                "Consider increasing novelty_per_fact parameter to encourage more fact sharing."
            )
        
        # Check if engagement dominates
        engagement_contrib = comp_stats.get("engagement", {}).get("contribution_pct", 0.0)
        if engagement_contrib > 70.0:
            recommendations.append(
                f"⚠️  Engagement reward contributes {engagement_contrib:.1f}% of total reward. "
                "This may cause the agent to prioritize engagement over content delivery."
            )
        
        # Note: Question spam is handled at simulator level (reduces dwell time)
        # No explicit penalty in reward function, so no check needed here
        
        return recommendations
    
    def _generate_action_recommendations(self, option_counts: Dict, subaction_counts: Dict, total: int) -> List[str]:
        """Generate recommendations based on action distribution"""
        recommendations = []
        
        explain_pct = (option_counts.get("Explain", 0) / total * 100) if total > 0 else 0.0
        ask_pct = (option_counts.get("AskQuestion", 0) / total * 100) if total > 0 else 0.0
        
        if explain_pct < 40.0:
            recommendations.append(
                f"⚠️  Agent uses Explain only {explain_pct:.1f}% of the time. "
                "Consider increasing novelty_per_fact parameter to encourage more fact sharing."
            )
        
        if ask_pct > 30.0:
            recommendations.append(
                f"⚠️  Agent uses AskQuestion {ask_pct:.1f}% of the time. "
                "This may indicate question spam. Consider increasing question spam penalty."
            )
        
        explain_new_fact_pct = (subaction_counts.get("ExplainNewFact", 0) / total * 100) if total > 0 else 0.0
        if explain_new_fact_pct < 20.0:
            recommendations.append(
                f"⚠️  Agent uses ExplainNewFact only {explain_new_fact_pct:.1f}% of the time. "
                "Consider increasing novelty_per_fact parameter to make fact sharing more rewarding."
            )
        
        return recommendations
    
    def _generate_novelty_recommendations(self, explain_facts: List[int], total_facts: List[int]) -> List[str]:
        """Generate recommendations based on novelty analysis"""
        recommendations = []
        
        if explain_facts:
            avg_facts = np.mean(explain_facts)
            if avg_facts < 1.0:
                recommendations.append(
                    f"⚠️  ExplainNewFact averages only {avg_facts:.2f} new facts per use. "
                    "Consider improving prompt or increasing reward per fact."
                )
        
        if total_facts:
            avg_total = np.mean(total_facts)
            if avg_total < 0.5:
                recommendations.append(
                    f"⚠️  Agent shares only {avg_total:.2f} new facts per turn on average. "
                    "Consider increasing novelty reward weight significantly."
                )
        
        return recommendations
    
    def generate_full_report(self, episodes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Generate comprehensive parameterization analysis report"""
        report = {
            "reward_analysis": self.analyze_reward_components(episodes),
            "action_analysis": self.analyze_action_distributions(episodes),
            "novelty_analysis": self.analyze_novelty_patterns(episodes)
        }
        
        # Save report
        report_file = self.results_dir / "parameterization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Generate text summary
        self._generate_text_summary(report)
        
        return report
    
    def _generate_text_summary(self, report: Dict[str, Any]):
        """Generate human-readable summary"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("PARAMETERIZATION ANALYSIS REPORT")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Reward analysis
        reward_analysis = report.get("reward_analysis", {})
        if "reward_component_stats" in reward_analysis:
            summary_lines.append("REWARD COMPONENT ANALYSIS")
            summary_lines.append("-" * 80)
            stats = reward_analysis["reward_component_stats"].get("reward_component_stats", {})
            for component, stat in stats.items():
                if isinstance(stat, dict) and "contribution_pct" in stat:
                    # Convert component to string in case it's an int
                    component_str = str(component)
                    summary_lines.append(
                        f"  {component_str:25s}: {stat['contribution_pct']:6.2f}% contribution "
                        f"(mean: {stat['mean']:+.3f}, sum: {stat['sum']:+.3f})"
                    )
            summary_lines.append("")
            
            if "recommendations" in reward_analysis:
                summary_lines.append("RECOMMENDATIONS:")
                for rec in reward_analysis["recommendations"]:
                    summary_lines.append(f"  {rec}")
                summary_lines.append("")
        
        # Action analysis
        action_analysis = report.get("action_analysis", {})
        if "option_distribution" in action_analysis:
            summary_lines.append("ACTION DISTRIBUTION")
            summary_lines.append("-" * 80)
            for option, data in action_analysis["option_distribution"].items():
                # Convert option to string in case it's an int
                option_str = str(option)
                summary_lines.append(
                    f"  {option_str:20s}: {data['count']:4d} uses ({data['percentage']:5.2f}%)"
                )
            summary_lines.append("")
            
            if "recommendations" in action_analysis:
                summary_lines.append("RECOMMENDATIONS:")
                for rec in action_analysis["recommendations"]:
                    summary_lines.append(f"  {rec}")
                summary_lines.append("")
        
        # Novelty analysis
        novelty_analysis = report.get("novelty_analysis", {})
        if "explain_new_fact_usage" in novelty_analysis:
            summary_lines.append("NOVELTY ANALYSIS")
            summary_lines.append("-" * 80)
            usage = novelty_analysis["explain_new_fact_usage"]
            summary_lines.append(f"  ExplainNewFact used: {usage['count']} times")
            summary_lines.append(f"  Avg facts per use: {usage['avg_facts_per_use']:.2f}")
            summary_lines.append(f"  Total facts introduced: {usage['total_facts_introduced']}")
            summary_lines.append("")
            
            if "recommendations" in novelty_analysis:
                summary_lines.append("RECOMMENDATIONS:")
                for rec in novelty_analysis["recommendations"]:
                    summary_lines.append(f"  {rec}")
        
        summary_lines.append("")
        summary_lines.append("=" * 80)
        
        summary_file = self.results_dir / "parameterization_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
