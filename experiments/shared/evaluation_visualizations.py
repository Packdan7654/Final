"""
Evaluation Visualizations for Hypothesis Testing

Generates all plots required for each hypothesis evaluation.
All plots are automatically saved to visualizations/evaluation/ directory.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class EvaluationPlotGenerator:
    """Generate evaluation plots for hypothesis testing."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory to save plots (usually visualizations/evaluation/)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== H1 PLOTS =====
    
    def plot_h1_comparison(self, baseline_metrics: Dict, h1_metrics: Dict):
        """
        Generate H1 comparison plots: option structure vs flat policy.
        
        Plots:
        - Strategy timeline (option usage over tour)
        - Mean return with variance comparison
        - Average coherent-span length comparison
        - Switch rate comparison
        - Learning curves
        """
        try:
            # 1. Mean return with variance comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            models = ['Baseline (Hierarchical)', 'H1 (Flat)']
            means = [
                baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                h1_metrics.get('common_metrics', {}).get('mean_return', 0.0)
            ]
            stds = [
                baseline_metrics.get('common_metrics', {}).get('std_return', 0.0),
                h1_metrics.get('common_metrics', {}).get('std_return', 0.0)
            ]
            
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Mean Episodic Return')
            ax.set_xlabel('Model')
            ax.set_title('H1: Mean Return with Variance Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.1, f'{mean:.2f}±{std:.2f}', 
                       ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h1_mean_return_comparison.png', bbox_inches='tight')
            plt.close()
            
            # 2. Coherent-span length comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            baseline_spans = baseline_metrics.get('h1_metrics', {}).get('coherent_span_lengths', [])
            h1_spans = h1_metrics.get('h1_metrics', {}).get('coherent_span_lengths', [])
            
            if baseline_spans and h1_spans:
                data = [baseline_spans, h1_spans]
                bp = ax.boxplot(data, labels=['Baseline', 'H1'], patch_artist=True)
                bp['boxes'][0].set_facecolor('steelblue')
                bp['boxes'][1].set_facecolor('coral')
                ax.set_ylabel('Coherent Span Length (turns)')
                ax.set_title('H1: Average Coherent-Span Length Comparison')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h1_coherent_span_comparison.png', bbox_inches='tight')
            plt.close()
            
            # 3. Switch rate comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            switch_rates = [
                baseline_metrics.get('h1_metrics', {}).get('switch_rate_per_100_turns', 0.0),
                h1_metrics.get('h1_metrics', {}).get('switch_rate_per_100_turns', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H1'], switch_rates, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Switch Rate (per 100 turns)')
            ax.set_title('H1: Switch Rate Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, rate in enumerate(switch_rates):
                ax.text(i, rate + 0.5, f'{rate:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h1_switch_rate_comparison.png', bbox_inches='tight')
            plt.close()
            
            logger.info("Generated H1 comparison plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H1 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H2 PLOTS =====
    
    def plot_h2_terminations(self, h2_metrics: Dict):
        """
        Generate H2 plots: learned terminations analysis.
        
        Plots:
        - Duration vs dwell scatter plot (with confidence bands)
        - Dwell correlation with confidence intervals
        - Time-to-termination comparison
        - Pre/post intent change windows
        """
        try:
            durations = h2_metrics.get('explain_durations', [])
            dwells = h2_metrics.get('dwell_during_explain', [])
            correlation = h2_metrics.get('dwell_correlation', 0.0)
            ci = h2_metrics.get('correlation_confidence_interval', {})
            
            if len(durations) > 1 and len(dwells) > 1:
                # 1. Duration vs dwell scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(durations, dwells, alpha=0.6, s=50)
                
                # Add regression line
                if len(durations) > 1:
                    z = np.polyfit(durations, dwells, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(durations), max(durations), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'ρ = {correlation:.3f}')
                
                ax.set_xlabel('Explain Segment Duration (τ, turns)')
                ax.set_ylabel('Mean Dwell During Segment')
                ax.set_title('H2: Duration vs Dwell Correlation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add confidence interval text
                if ci:
                    ci_text = f"95% CI: [{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]"
                    ax.text(0.05, 0.95, ci_text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h2_duration_vs_dwell.png', bbox_inches='tight')
                plt.close()
                
                # 2. Time-to-termination comparison
                term_after_intent = h2_metrics.get('termination_after_intent_change', [])
                term_no_intent = h2_metrics.get('termination_no_intent_change', [])
                
                if term_after_intent and term_no_intent:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data = [term_after_intent, term_no_intent]
                    bp = ax.boxplot(data, labels=['After Intent Change', 'No Intent Change'],
                                   patch_artist=True)
                    bp['boxes'][0].set_facecolor('coral')
                    bp['boxes'][1].set_facecolor('steelblue')
                    ax.set_ylabel('Time to Termination (turns)')
                    ax.set_title('H2: Termination Timing Comparison')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'h2_termination_timing.png', bbox_inches='tight')
                    plt.close()
            
            logger.info("Generated H2 termination plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H2 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H3 PLOTS =====
    
    def plot_h3_grounding(self, baseline_metrics: Dict, h3_metrics: Dict):
        """
        Generate H3 plots: prompt headers comparison.
        
        Plots:
        - Grounding precision/recall comparison
        - Hallucination rate over training
        - On-plan compliance rates by subaction
        - Novel-fact coverage evolution
        - Act classifier agreement scores
        - Header completeness vs errors correlation
        """
        try:
            # 1. Grounding precision/recall comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            models = ['Baseline', 'H3']
            precision = [
                baseline_metrics.get('h3_metrics', {}).get('grounding_precision', 0.0),
                h3_metrics.get('h3_metrics', {}).get('grounding_precision', 0.0)
            ]
            recall = [
                baseline_metrics.get('h3_metrics', {}).get('grounding_recall', 0.0),
                h3_metrics.get('h3_metrics', {}).get('grounding_recall', 0.0)
            ]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, precision, width, label='Precision', alpha=0.7, color='steelblue')
            bars2 = ax.bar(x + width/2, recall, width, label='Recall', alpha=0.7, color='coral')
            
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.set_title('H3: Grounding Precision/Recall Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h3_grounding_comparison.png', bbox_inches='tight')
            plt.close()
            
            # 2. Hallucination rate comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            hallucination_rates = [
                baseline_metrics.get('h3_metrics', {}).get('hallucination_rate', 0.0),
                h3_metrics.get('h3_metrics', {}).get('hallucination_rate', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H3'], hallucination_rates, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Hallucination Rate')
            ax.set_title('H3: Hallucination Rate Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, rate in enumerate(hallucination_rates):
                ax.text(i, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h3_hallucination_rate.png', bbox_inches='tight')
            plt.close()
            
            # 3. On-plan compliance by subaction
            baseline_compliance = baseline_metrics.get('h3_metrics', {}).get('on_plan_compliance', {})
            h3_compliance = h3_metrics.get('h3_metrics', {}).get('on_plan_compliance', {})
            
            if baseline_compliance or h3_compliance:
                fig, ax = plt.subplots(figsize=(12, 6))
                subactions = sorted(set(list(baseline_compliance.keys()) + list(h3_compliance.keys())))
                baseline_rates = [baseline_compliance.get(s, {}).get('compliance_rate', 0.0) for s in subactions]
                h3_rates = [h3_compliance.get(s, {}).get('compliance_rate', 0.0) for s in subactions]
                
                x = np.arange(len(subactions))
                width = 0.35
                
                ax.bar(x - width/2, baseline_rates, width, label='Baseline', alpha=0.7, color='steelblue')
                ax.bar(x + width/2, h3_rates, width, label='H3', alpha=0.7, color='coral')
                
                ax.set_ylabel('Compliance Rate')
                ax.set_xlabel('Subaction')
                ax.set_title('H3: On-Plan Compliance by Subaction')
                ax.set_xticks(x)
                ax.set_xticklabels(subactions, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([0, 1.1])
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h3_on_plan_compliance.png', bbox_inches='tight')
                plt.close()
            
            logger.info("Generated H3 grounding plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H3 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H4 PLOTS =====
    
    def plot_h4_stability(self, baseline_metrics: Dict, h1_metrics: Dict,
                         baseline_training_logs: Dict, h1_training_logs: Dict):
        """
        Generate H4 plots: training stability comparison.
        
        Plots:
        - Learning curves: Smoothed return vs updates (baseline vs H1)
        - Update magnitude comparison: L2-norm of parameter steps, KL per update
        - Time-to-target comparison
        - Option duration sanity
        """
        try:
            # 1. Learning curves comparison (smoothed return vs updates)
            baseline_returns = baseline_training_logs.get('episode_returns', [])
            h1_returns = h1_training_logs.get('episode_returns', [])
            
            if baseline_returns and h1_returns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Smooth the curves
                window = 50
                baseline_smooth = self._moving_average(baseline_returns, window)
                h1_smooth = self._moving_average(h1_returns, window)
                
                episodes = np.arange(1, len(baseline_smooth) + 1)
                ax.plot(episodes, baseline_smooth, label='Baseline (Hierarchical)', 
                       linewidth=2, color='steelblue')
                ax.plot(episodes[:len(h1_smooth)], h1_smooth, label='H1 (Flat)', 
                       linewidth=2, color='coral')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Smoothed Return')
                ax.set_title('H4: Learning Curve Comparison (Smoothed)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h4_learning_curves.png', bbox_inches='tight')
                plt.close()
            
            # 2. Update magnitude comparison
            baseline_updates = baseline_training_logs.get('update_norms', [])
            h1_updates = h1_training_logs.get('update_norms', [])
            
            if baseline_updates and h1_updates:
                fig, ax = plt.subplots(figsize=(10, 6))
                data = [baseline_updates, h1_updates]
                bp = ax.boxplot(data, labels=['Baseline', 'H1'], patch_artist=True)
                bp['boxes'][0].set_facecolor('steelblue')
                bp['boxes'][1].set_facecolor('coral')
                ax.set_ylabel('Update Magnitude (L2-norm)')
                ax.set_title('H4: Update Magnitude Comparison')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h4_update_magnitudes.png', bbox_inches='tight')
                plt.close()
            
            logger.info("Generated H4 stability plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H4 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H5 PLOTS =====
    
    def plot_h5_state_ablation(self, baseline_metrics: Dict, h5_metrics: Dict):
        """
        Generate H5 plots: state ablation comparison.
        
        Plots:
        - State dimension comparison
        - Dialogue act distribution histograms
        - Dialogue act probability entropy over time
        - Act consistency scores
        - Responsiveness rate comparison
        - Dwell time comparison
        """
        try:
            # 1. State dimension comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            dimensions = [
                baseline_metrics.get('common_metrics', {}).get('state_dimension', 149),
                h5_metrics.get('h5_metrics', {}).get('state_dimension', 23)
            ]
            
            bars = ax.bar(['Baseline', 'H5'], dimensions, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('State Dimension')
            ax.set_title('H5: State Dimension Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, dim in enumerate(dimensions):
                ax.text(i, dim + 2, f'{dim}-d', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h5_state_dimension.png', bbox_inches='tight')
            plt.close()
            
            # 2. Dialogue act distribution
            h5_dist = h5_metrics.get('h5_metrics', {}).get('dialogue_act_distribution', {})
            if h5_dist:
                fig, ax = plt.subplots(figsize=(10, 6))
                acts = list(h5_dist.keys())
                probs = list(h5_dist.values())
                
                bars = ax.bar(acts, probs, alpha=0.7, color='steelblue')
                ax.set_ylabel('Probability')
                ax.set_xlabel('Dialogue Act')
                ax.set_title('H5: Dialogue Act Distribution (8 categories)')
                ax.set_xticklabels(acts, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h5_dialogue_act_distribution.png', bbox_inches='tight')
                plt.close()
            
            # 3. Act consistency comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            consistency = [
                baseline_metrics.get('h5_metrics', {}).get('act_consistency', 0.0),
                h5_metrics.get('h5_metrics', {}).get('act_consistency', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H5'], consistency, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Act Consistency Score')
            ax.set_title('H5: Act Consistency Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])
            
            for i, score in enumerate(consistency):
                ax.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h5_act_consistency.png', bbox_inches='tight')
            plt.close()
            
            logger.info("Generated H5 state ablation plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H5 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H6 PLOTS =====
    
    def plot_h6_transitions(self, baseline_metrics: Dict, h6_metrics: Dict):
        """
        Generate H6 plots: transition reward comparison.
        
        Plots:
        - Transition timing distribution
        - Transition success rate comparison
        - Premature transition rate comparison
        - Facts before transition box plots
        - Confusion rate after transitions
        - Reward component breakdown
        """
        try:
            # 1. Transition success rate comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            success_rates = [
                baseline_metrics.get('h6_metrics', {}).get('transition_success_rate', 0.0),
                h6_metrics.get('h6_metrics', {}).get('transition_success_rate', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H6'], success_rates, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Transition Success Rate')
            ax.set_title('H6: Transition Success Rate Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])
            
            for i, rate in enumerate(success_rates):
                ax.text(i, rate + 0.02, f'{rate:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h6_transition_success.png', bbox_inches='tight')
            plt.close()
            
            # 2. Premature transition rate comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            premature_rates = [
                baseline_metrics.get('h6_metrics', {}).get('premature_transition_rate', 0.0),
                h6_metrics.get('h6_metrics', {}).get('premature_transition_rate', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H6'], premature_rates, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Premature Transition Rate')
            ax.set_title('H6: Premature Transition Rate Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, rate in enumerate(premature_rates):
                ax.text(i, rate + 0.01, f'{rate:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h6_premature_transitions.png', bbox_inches='tight')
            plt.close()
            
            logger.info("Generated H6 transition plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H6 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== H7 PLOTS =====
    
    def plot_h7_coherence(self, baseline_metrics: Dict, h7_metrics: Dict):
        """
        Generate H7 plots: hybrid BERT comparison.
        
        Plots:
        - Dialogue coherence scores
        - Context-dependent QA accuracy
        - Embedding similarity over dialogue turns
        - Reference resolution accuracy
        """
        try:
            # 1. Dialogue coherence comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            coherence = [
                baseline_metrics.get('h7_metrics', {}).get('dialogue_coherence', 0.0),
                h7_metrics.get('h7_metrics', {}).get('dialogue_coherence', 0.0)
            ]
            
            bars = ax.bar(['Baseline', 'H7'], coherence, alpha=0.7,
                         color=['steelblue', 'coral'])
            ax.set_ylabel('Dialogue Coherence Score')
            ax.set_title('H7: Dialogue Coherence Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])
            
            for i, score in enumerate(coherence):
                ax.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'h7_coherence.png', bbox_inches='tight')
            plt.close()
            
            # 2. Embedding similarity
            h7_similarities = h7_metrics.get('h7_metrics', {}).get('embedding_similarity', [])
            if h7_similarities:
                fig, ax = plt.subplots(figsize=(10, 6))
                turns = np.arange(1, len(h7_similarities) + 1)
                ax.plot(turns, h7_similarities, linewidth=2, color='steelblue', alpha=0.7)
                ax.set_xlabel('Dialogue Turn')
                ax.set_ylabel('Embedding Similarity (Cosine)')
                ax.set_title('H7: Embedding Similarity Over Dialogue Turns')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'h7_embedding_similarity.png', bbox_inches='tight')
                plt.close()
            
            logger.info("Generated H7 coherence plots")
            
        except Exception as e:
            logger.error(f"Failed to generate H7 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== HELPER METHODS =====
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Compute moving average."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def load_metrics_from_file(self, metrics_file: Path) -> Dict:
        """Load metrics from JSON file."""
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_file}: {e}")
            return {}


def generate_hypothesis_plots(hypothesis: str, metrics: Dict, output_dir: Path,
                             baseline_metrics: Optional[Dict] = None,
                             comparison_metrics: Optional[Dict] = None,
                             training_logs: Optional[Dict] = None):
    """
    Generate all plots for a specific hypothesis.
    
    Args:
        hypothesis: Hypothesis identifier ('h1', 'h2', 'h3', 'h5', 'h6', 'h7')
        metrics: Metrics dictionary for the hypothesis
        output_dir: Directory to save plots
        baseline_metrics: Baseline metrics (for comparisons)
        comparison_metrics: Comparison model metrics (H1, H3, H5, H6, H7)
        training_logs: Training logs (for H4)
    """
    generator = EvaluationPlotGenerator(output_dir)
    
    try:
        if hypothesis == 'h1':
            if baseline_metrics and comparison_metrics:
                generator.plot_h1_comparison(baseline_metrics, comparison_metrics)
            else:
                logger.warning("H1 plots require baseline and H1 metrics - skipping comparison plots")
        elif hypothesis == 'h2':
            generator.plot_h2_terminations(metrics)
        elif hypothesis == 'h3':
            if baseline_metrics and comparison_metrics:
                generator.plot_h3_grounding(baseline_metrics, comparison_metrics)
            else:
                logger.warning("H3 plots require baseline and H3 metrics - skipping comparison plots")
        elif hypothesis == 'h4':
            if baseline_metrics and comparison_metrics and training_logs:
                baseline_logs = training_logs.get('baseline', {})
                h1_logs = training_logs.get('h1', {})
                generator.plot_h4_stability(baseline_metrics, comparison_metrics,
                                           baseline_logs, h1_logs)
            else:
                logger.warning("H4 plots require baseline, H1 metrics, and training logs - skipping plots")
        elif hypothesis == 'h5':
            if baseline_metrics and comparison_metrics:
                generator.plot_h5_state_ablation(baseline_metrics, comparison_metrics)
            else:
                logger.warning("H5 plots require baseline and H5 metrics - skipping comparison plots")
        elif hypothesis == 'h6':
            if baseline_metrics and comparison_metrics:
                generator.plot_h6_transitions(baseline_metrics, comparison_metrics)
            else:
                logger.warning("H6 plots require baseline and H6 metrics - skipping comparison plots")
        elif hypothesis == 'h7':
            if baseline_metrics and comparison_metrics:
                generator.plot_h7_coherence(baseline_metrics, comparison_metrics)
            else:
                logger.warning("H7 plots require baseline and H7 metrics - skipping comparison plots")
    except Exception as e:
        logger.error(f"Failed to generate plots for {hypothesis}: {e}")
        import traceback
        traceback.print_exc()

