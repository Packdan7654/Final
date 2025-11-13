"""
Shared Evaluation Framework for Hypothesis Testing

Provides standardized evaluation metrics and comparison tools
for all experimental variants (baseline, H5, H6).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from scipy import stats
from scipy.stats import bootstrap
import logging

logger = logging.getLogger(__name__)


class HypothesisEvaluator:
    """
    Evaluator for hypothesis-specific metrics.
    
    Handles:
    - H5: Dialogue act state coherence, state dimension comparison
    - H6: Transition success rates, reward component analysis
    - H7: Dialogue coherence, context-dependent QA, embedding similarity
    - Common: Return, coverage, dwell, episode length
    """
    
    def __init__(self, experiment_name: str, results_dir: Path):
        """
        Initialize evaluator.
        
        Args:
            experiment_name: Name of experiment variant (baseline, h5, h6, h7)
            results_dir: Directory containing experiment results
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.episode_metrics = []
        self.turn_metrics = []
        self.experiment_path = None  # Store loaded experiment path
    
    def load_experiment_data(self, experiment_path: Path):
        """
        Load metrics from experiment directory.
        
        Args:
            experiment_path: Path to experiment results directory
        """
        exp_path = Path(experiment_path)
        self.experiment_path = exp_path  # Store for H7 metrics
        
        # Load metrics tracker JSON
        metrics_files = list((exp_path / 'logs').glob('metrics_tracker_*.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}
        
        # Load turn data
        turn_files = list((exp_path / 'logs').glob('monitor_*_turns_*.json'))
        if turn_files:
            with open(turn_files[0], 'r') as f:
                self.turn_data = json.load(f)
        else:
            self.turn_data = []
        
        # Load episode data
        episode_files = list((exp_path / 'logs').glob('monitor_*_episodes_*.json'))
        if episode_files:
            with open(episode_files[0], 'r') as f:
                self.episode_data = json.load(f)
        else:
            self.episode_data = []
    
    def compute_h5_metrics(self) -> Dict[str, Any]:
        """
        Compute H5-specific metrics (state ablation).
        
        Returns:
            Dictionary of H5 metrics:
            - state_dimension: State vector dimension
            - dialogue_act_distribution: Distribution of dialogue acts
            - state_compression_ratio: Reduction vs full DialogueBERT
            - act_consistency: Agreement between predicted option/subaction and realized act
            - responsiveness_rate: Fraction of visitor questions answered with new facts
            - mean_dwell: Average dwell time per episode
        """
        from experiments.shared.dialogue_act_classifier import get_dialogue_act_classifier
        
        metrics = {
            'state_dimension': None,
            'dialogue_act_distribution': {},
            'state_compression_ratio': 0.0,
            'act_consistency': 0.0,
            'responsiveness_rate': 0.0,
            'mean_dwell': 0.0,
            'act_consistency_by_option': {},
        }
        
        act_classifier = get_dialogue_act_classifier()
        
        # Extract state dimension from metadata if available
        metadata_file = self.results_dir.parent / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'state_dimension' in metadata:
                    metrics['state_dimension'] = metadata['state_dimension']
        
        # Count dialogue act types from turn data
        act_counts = defaultdict(int)
        act_consistencies = []
        act_consistency_by_option = defaultdict(list)
        
        # Responsiveness tracking
        total_questions = 0
        questions_answered = 0
        
        # Dwell tracking
        dwell_times = []
        
        for turn in self.turn_data:
            if 'dialogue_act' in turn:
                act_counts[turn['dialogue_act']] += 1
            
            # Act consistency: compare predicted option/subaction to realized utterance
            option = turn.get('option', 'Unknown')
            subaction = turn.get('subaction', 'Unknown')
            agent_utterance = turn.get('agent_utterance', '')
            user_utterance = turn.get('user_utterance', '')
            
            if agent_utterance:
                # Classify realized act
                realized_act = act_classifier.classify(agent_utterance)
                
                # Expected act based on option/subaction
                if option == 'AskQuestion':
                    expected_act = 'question'
                elif option == 'Explain':
                    expected_act = 'statement'  # Explanations are statements
                elif option == 'OfferTransition':
                    expected_act = 'statement'  # Transitions are statements
                elif option == 'Conclude':
                    expected_act = 'statement'  # Conclusions are statements
                else:
                    expected_act = 'statement'
                
                consistency = (realized_act == expected_act)
                act_consistencies.append(1 if consistency else 0)
                act_consistency_by_option[option].append(1 if consistency else 0)
            
            # Responsiveness: visitor questions answered with new facts
            if turn.get('response_type') == 'question' or '?' in (user_utterance or ''):
                total_questions += 1
                facts_mentioned = turn.get('facts_mentioned_in_utterance', [])
                if facts_mentioned:
                    questions_answered += 1
            
            # Dwell time
            dwell = turn.get('dwell', 0.0)
            if dwell > 0:
                dwell_times.append(dwell)
        
        total_acts = sum(act_counts.values())
        if total_acts > 0:
            metrics['dialogue_act_distribution'] = {
                act: count / total_acts 
                for act, count in act_counts.items()
            }
        
        # State compression: dialogue act (6-d) vs DialogueBERT (128-d)
        if metrics['state_dimension']:
            baseline_dim = 149  # Full DialogueBERT state
            compression = 1.0 - (metrics['state_dimension'] / baseline_dim)
            metrics['state_compression_ratio'] = float(compression)
        
        # Act consistency
        if act_consistencies:
            metrics['act_consistency'] = float(np.mean(act_consistencies))
            # Per-option consistency
            for opt, consistencies in act_consistency_by_option.items():
                if consistencies:
                    metrics['act_consistency_by_option'][opt] = float(np.mean(consistencies))
        
        # Responsiveness rate
        if total_questions > 0:
            metrics['responsiveness_rate'] = float(questions_answered / total_questions)
        
        # Mean dwell
        if dwell_times:
            metrics['mean_dwell'] = float(np.mean(dwell_times))
            metrics['median_dwell'] = float(np.median(dwell_times))
        
        return metrics
    
    def compute_h6_metrics(self) -> Dict[str, Any]:
        """
        Compute H6-specific metrics (transition reward shaping).
        
        Returns:
            Dictionary of H6 metrics:
            - transition_attempts: Total transition attempts
            - transition_success_rate: Success rate with/without shaping
            - avg_facts_before_transition: Average facts before transition
            - mean_dwell_before_transition: Mean dwell in turn preceding transition
            - confusion_after_transition: Confusion responses after transitions
            - premature_transition_rate: Transitions with < 2 facts
            - transition_timing_distribution: Histogram of facts before transition
            - reward_component_breakdown: Reward by component
        """
        metrics = {
            'transition_attempts': 0,
            'transition_successes': 0,
            'transition_success_rate': 0.0,
            'avg_facts_before_transition': 0.0,
            'mean_dwell_before_transition': 0.0,
            'confusion_after_transition': 0,
            'confusion_rate_after_transition': 0.0,
            'premature_transition_rate': 0.0,
            'transition_timing_distribution': {},
            'reward_component_breakdown': {},
        }
        
        # Extract from turn data
        transition_attempts = []
        transition_successes = []
        facts_before_transition = []
        dwell_before_transition = []
        confusion_after_transition = []
        premature_transitions = 0
        
        for i, turn in enumerate(self.turn_data):
            if turn.get('option') == 'OfferTransition':
                metrics['transition_attempts'] += 1
                transition_attempts.append(1)
                
                if turn.get('transition_success', False):
                    metrics['transition_successes'] += 1
                    transition_successes.append(1)
                else:
                    transition_successes.append(0)
                
                # Extract facts before transition
                facts_shared = turn.get('facts_shared', 0)
                if isinstance(facts_shared, (list, tuple)):
                    facts_shared = len(facts_shared)
                facts_before_transition.append(facts_shared)
                
                # Check for premature transition (< 2 facts)
                if facts_shared < 2:
                    premature_transitions += 1
                
                # Dwell in turn preceding transition (current turn's dwell)
                dwell = turn.get('dwell', 0.0)
                if dwell > 0:
                    dwell_before_transition.append(dwell)
                
                # Check for confusion in next turn (after transition)
                if i + 1 < len(self.turn_data):
                    next_turn = self.turn_data[i + 1]
                    response_type = next_turn.get('response_type', '')
                    if response_type == 'confusion':
                        confusion_after_transition.append(1)
                    else:
                        confusion_after_transition.append(0)
        
        if metrics['transition_attempts'] > 0:
            metrics['transition_success_rate'] = float(
                metrics['transition_successes'] / metrics['transition_attempts']
            )
            metrics['premature_transition_rate'] = float(
                premature_transitions / metrics['transition_attempts']
            )
        
        if facts_before_transition:
            metrics['avg_facts_before_transition'] = float(np.mean(facts_before_transition))
            metrics['median_facts_before_transition'] = float(np.median(facts_before_transition))
            metrics['min_facts_before_transition'] = float(np.min(facts_before_transition))
            metrics['max_facts_before_transition'] = float(np.max(facts_before_transition))
            
            # Transition timing distribution (histogram)
            fact_bins = [0, 1, 2, 3, 4, 5, 10, 20]  # Bins for facts
            hist, bin_edges = np.histogram(facts_before_transition, bins=fact_bins)
            metrics['transition_timing_distribution'] = {
                f'{int(bin_edges[i])}-{int(bin_edges[i+1])}': int(count)
                for i, count in enumerate(hist)
            }
        
        if dwell_before_transition:
            metrics['mean_dwell_before_transition'] = float(np.mean(dwell_before_transition))
            metrics['median_dwell_before_transition'] = float(np.median(dwell_before_transition))
        
        if confusion_after_transition:
            metrics['confusion_after_transition'] = int(np.sum(confusion_after_transition))
            metrics['confusion_rate_after_transition'] = float(
                np.mean(confusion_after_transition)
            )
        
        # Reward component breakdown
        reward_components = defaultdict(list)
        for turn in self.turn_data:
            for component in ['engagement', 'novelty', 'responsiveness', 
                            'transition', 'conclude']:
                key = f'reward_{component}'
                if key in turn:
                    reward_components[component].append(turn[key])
        
        metrics['reward_component_breakdown'] = {
            comp: {
                'total': float(np.sum(vals)),
                'mean': float(np.mean(vals)) if vals else 0.0,
                'std': float(np.std(vals)) if vals else 0.0,
            }
            for comp, vals in reward_components.items()
        }
        
        return metrics
    
    def compute_common_metrics(self) -> Dict[str, Any]:
        """
        Compute common metrics for all experiments.
        
        Returns:
            Dictionary of common metrics including dwell averages
        """
        returns = self.metrics.get('episode_returns', [])
        lengths = self.metrics.get('episode_lengths', [])
        coverage = self.metrics.get('episode_coverage', [])
        
        # Extract dwell times from turn data
        dwell_times = [turn.get('dwell', 0.0) for turn in self.turn_data if turn.get('dwell', 0.0) > 0]
        
        metrics = {
            'mean_return': float(np.mean(returns)) if returns else 0.0,
            'std_return': float(np.std(returns)) if returns else 0.0,
            'mean_length': float(np.mean(lengths)) if lengths else 0.0,
            'mean_coverage': float(np.mean(coverage)) if coverage else 0.0,
            'total_episodes': len(returns),
            'mean_dwell': float(np.mean(dwell_times)) if dwell_times else 0.0,
            'median_dwell': float(np.median(dwell_times)) if dwell_times else 0.0,
        }
        
        return metrics
    
    def compute_h1_metrics(self) -> Dict[str, Any]:
        """
        Compute H1-specific metrics (option structure vs flat).
        
        Returns:
            Dictionary of H1 metrics with all standard HRL metrics:
            - coherent_span_lengths: Consecutive turns under same option
            - switch_rate: Option switches per 100 turns
            - option_persistence: Average duration per option
            - option_duration_stats: Duration statistics per option (median, IQR)
            - option_usage_frequency: Usage frequency per option
            - option_transition_matrix: Transition probabilities between options
            - policy_entropy: Entropy of option policy (if available)
        """
        metrics = {
            'coherent_span_lengths': [],
            'switch_rate_per_100_turns': 0.0,
            'option_persistence': {},
            'option_duration_stats': {},
            'option_usage_frequency': {},
            'option_transition_matrix': {},
            'policy_entropy': None,
        }
        
        current_option = None
        span_length = 0
        total_switches = 0
        total_turns = len(self.turn_data)
        option_durations = defaultdict(list)
        option_counts = defaultdict(int)
        transitions = defaultdict(lambda: defaultdict(int))  # from_option -> to_option -> count
        
        for i, turn in enumerate(self.turn_data):
            option = turn.get('option', 'Unknown')
            option_counts[option] += 1
            
            if option != current_option:
                if current_option is not None:
                    total_switches += 1
                    if span_length > 0:
                        metrics['coherent_span_lengths'].append(span_length)
                        option_durations[current_option].append(span_length)
                        if current_option not in metrics['option_persistence']:
                            metrics['option_persistence'][current_option] = []
                        metrics['option_persistence'][current_option].append(span_length)
                        # Track transition
                        transitions[current_option][option] += 1
                current_option = option
                span_length = 1
            else:
                span_length += 1
        
        if span_length > 0 and current_option:
            metrics['coherent_span_lengths'].append(span_length)
            option_durations[current_option].append(span_length)
            if current_option not in metrics['option_persistence']:
                metrics['option_persistence'][current_option] = []
            metrics['option_persistence'][current_option].append(span_length)
        
        if total_turns > 0:
            metrics['switch_rate_per_100_turns'] = (total_switches / total_turns) * 100
        
        if metrics['coherent_span_lengths']:
            metrics['mean_coherent_span'] = np.mean(metrics['coherent_span_lengths'])
            metrics['median_coherent_span'] = np.median(metrics['coherent_span_lengths'])
            metrics['std_coherent_span'] = np.std(metrics['coherent_span_lengths'])
            # IQR
            q25, q75 = np.percentile(metrics['coherent_span_lengths'], [25, 75])
            metrics['iqr_coherent_span'] = q75 - q25
        else:
            metrics['mean_coherent_span'] = 0.0
            metrics['median_coherent_span'] = 0.0
            metrics['std_coherent_span'] = 0.0
            metrics['iqr_coherent_span'] = 0.0
        
        # Option duration statistics (median, IQR per option)
        for opt, durations in option_durations.items():
            if durations:
                metrics['option_duration_stats'][opt] = {
                    'median': float(np.median(durations)),
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'q25': float(np.percentile(durations, 25)),
                    'q75': float(np.percentile(durations, 75)),
                    'iqr': float(np.percentile(durations, 75) - np.percentile(durations, 25)),
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                }
        
        # Option usage frequency
        total_option_uses = sum(option_counts.values())
        if total_option_uses > 0:
            metrics['option_usage_frequency'] = {
                opt: count / total_option_uses
                for opt, count in option_counts.items()
            }
        
        # Option transition matrix (normalized probabilities)
        all_options = set(option_counts.keys())
        for from_opt in all_options:
            total_from = sum(transitions[from_opt].values())
            if total_from > 0:
                metrics['option_transition_matrix'][from_opt] = {
                    to_opt: count / total_from
                    for to_opt, count in transitions[from_opt].items()
                }
            else:
                metrics['option_transition_matrix'][from_opt] = {}
        
        # Policy entropy (if available in metrics)
        if 'option_usage_frequency' in self.metrics:
            usage_probs = list(metrics['option_usage_frequency'].values())
            if usage_probs and sum(usage_probs) > 0:
                # Normalize
                usage_probs = np.array(usage_probs) / sum(usage_probs)
                # Compute entropy: -sum(p * log(p))
                entropy = -np.sum(usage_probs * np.log(usage_probs + 1e-10))
                metrics['policy_entropy'] = float(entropy)
        
        return metrics
    
    def compute_h2_metrics(self) -> Dict[str, Any]:
        """
        Compute H2-specific metrics (learned terminations).
        
        Returns:
            Dictionary of H2 metrics:
            - explain_durations: Duration of Explain option segments
            - dwell_correlation: Correlation between dwell and Explain duration
            - termination_timing: Timing of terminations relative to intent changes
            - intent_change_detection: Detected intent changes
            - termination_rate: Terminations per option instance
            - correlation_confidence_interval: CI for dwell correlation
        """
        metrics = {
            'explain_durations': [],
            'dwell_during_explain': [],
            'dwell_correlation': 0.0,
            'correlation_confidence_interval': None,
            'intent_changes': [],
            'termination_after_intent_change': [],
            'termination_no_intent_change': [],
            'termination_rate': 0.0,
            'mean_explain_duration': 0.0,
        }
        
        current_explain_start = None
        explain_dwells = []
        previous_response_type = None
        intent_changes = []
        
        # Detect intent changes (shift away from "explain-supporting" responses)
        explain_supporting_types = ['acknowledgment', 'follow_up_question']
        non_explain_types = ['question', 'confusion', 'statement']
        
        for i, turn in enumerate(self.turn_data):
            option = turn.get('option', 'Unknown')
            dwell = turn.get('dwell', 0.0)
            response_type = turn.get('response_type', 'statement')
            
            # Detect intent change: shift from explain-supporting to non-explain
            if previous_response_type in explain_supporting_types and response_type in non_explain_types:
                intent_changes.append(i)
            
            previous_response_type = response_type
            
            if option == 'Explain':
                if current_explain_start is None:
                    current_explain_start = i
                    explain_dwells = []
                explain_dwells.append(dwell)
            else:
                if current_explain_start is not None:
                    duration = i - current_explain_start
                    metrics['explain_durations'].append(duration)
                    if explain_dwells:
                        metrics['dwell_during_explain'].append(np.mean(explain_dwells))
                    
                    # Check if termination happened after intent change
                    # Look for intent changes in the last 3 turns before termination
                    recent_intent_changes = [ic for ic in intent_changes if current_explain_start <= ic < i]
                    if recent_intent_changes:
                        # Termination after intent change
                        time_to_termination = i - max(recent_intent_changes)
                        metrics['termination_after_intent_change'].append(time_to_termination)
                    else:
                        # Termination without intent change
                        metrics['termination_no_intent_change'].append(duration)
                    
                    current_explain_start = None
                    explain_dwells = []
        
        # Final segment
        if current_explain_start is not None:
            duration = len(self.turn_data) - current_explain_start
            metrics['explain_durations'].append(duration)
            if explain_dwells:
                metrics['dwell_during_explain'].append(np.mean(explain_dwells))
        
        metrics['intent_changes'] = intent_changes
        
        # Correlation with confidence interval
        if len(metrics['explain_durations']) > 1 and len(metrics['dwell_during_explain']) > 1:
            if len(metrics['explain_durations']) == len(metrics['dwell_during_explain']):
                corr = np.corrcoef(
                    metrics['explain_durations'],
                    metrics['dwell_during_explain']
                )[0, 1] if len(metrics['explain_durations']) > 1 else 0.0
                metrics['dwell_correlation'] = float(corr)
                
                # Compute confidence interval using bootstrap
                try:
                    def correlation_statistic(data):
                        durations, dwells = data
                        if len(durations) < 2:
                            return 0.0
                        corr = np.corrcoef(durations, dwells)[0, 1]
                        return corr if not np.isnan(corr) else 0.0
                    
                    data = (np.array(metrics['explain_durations']), 
                           np.array(metrics['dwell_during_explain']))
                    ci_result = bootstrap((data,), correlation_statistic, n_resamples=1000, 
                                         confidence_level=0.95, method='percentile')
                    metrics['correlation_confidence_interval'] = {
                        'lower': float(ci_result.confidence_interval.low),
                        'upper': float(ci_result.confidence_interval.high),
                    }
                except:
                    # Fallback: approximate CI using Fisher transformation
                    n = len(metrics['explain_durations'])
                    if n > 3 and abs(corr) < 0.99:
                        z = 0.5 * np.log((1 + corr) / (1 - corr))
                        se = 1.0 / np.sqrt(n - 3)
                        z_lower = z - 1.96 * se
                        z_upper = z + 1.96 * se
                        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                        metrics['correlation_confidence_interval'] = {
                            'lower': float(ci_lower),
                            'upper': float(ci_upper),
                        }
        
        if metrics['explain_durations']:
            metrics['mean_explain_duration'] = float(np.mean(metrics['explain_durations']))
            metrics['median_explain_duration'] = float(np.median(metrics['explain_durations']))
        else:
            metrics['mean_explain_duration'] = 0.0
            metrics['median_explain_duration'] = 0.0
        
        # Termination rate (terminations per Explain instance)
        total_explain_instances = len(metrics['explain_durations'])
        if total_explain_instances > 0:
            # Count actual terminations (segments that ended before episode end)
            # This is approximate - we assume all segments except the last are terminations
            terminations = max(0, total_explain_instances - 1)
            metrics['termination_rate'] = terminations / total_explain_instances if total_explain_instances > 0 else 0.0
        
        # Compare termination timing
        if metrics['termination_after_intent_change'] and metrics['termination_no_intent_change']:
            metrics['mean_termination_after_intent_change'] = float(np.mean(metrics['termination_after_intent_change']))
            metrics['mean_termination_no_intent_change'] = float(np.mean(metrics['termination_no_intent_change']))
            metrics['termination_timing_difference'] = (
                metrics['mean_termination_after_intent_change'] - 
                metrics['mean_termination_no_intent_change']
            )
        
        return metrics
    
    def compute_h3_metrics(self) -> Dict[str, Any]:
        """
        Compute H3-specific metrics (prompt headers).
        
        Returns:
            Dictionary of H3 metrics:
            - novel_fact_coverage: Coverage of novel facts
            - repetition_ratio: Ratio of repeated facts
            - grounding_precision: Precision of KB grounding
            - grounding_recall: Recall of KB grounding
            - hallucination_rate: Rate of non-KB claims
            - on_plan_compliance: Agreement between intended and realized acts
            - act_classifier_agreement: Agreement with dialogue act classifier
            - header_completeness: Average header completeness score
        """
        from experiments.shared.dialogue_act_classifier import get_dialogue_act_classifier
        
        metrics = {
            'novel_fact_coverage': 0.0,
            'repetition_ratio': 0.0,
            'grounding_precision': 0.0,
            'grounding_recall': 0.0,
            'kb_citation_rate': 0.0,
            'hallucination_rate': 0.0,
            'on_plan_compliance': {},
            'act_classifier_agreement': 0.0,
            'header_completeness': 0.0,
            'header_completeness_vs_errors': None,
        }
        
        act_classifier = get_dialogue_act_classifier()
        
        total_facts_mentioned = 0
        unique_facts = set()
        fact_mentions = {}
        kb_citations = 0
        hallucinations = 0
        
        # Grounding metrics
        true_positives = 0  # Valid KB facts mentioned
        false_positives = 0  # Hallucinated facts
        false_negatives = 0  # Valid facts not mentioned (approximate)
        
        # On-plan compliance tracking
        on_plan_violations = defaultdict(int)
        on_plan_compliant = defaultdict(int)
        
        # Act classifier agreement
        act_agreements = []
        header_completeness_scores = []
        completeness_vs_errors = []
        
        for turn in self.turn_data:
            option = turn.get('option', 'Unknown')
            subaction = turn.get('subaction', 'Unknown')
            facts = turn.get('facts_mentioned_in_utterance', [])
            hallucinated = turn.get('hallucinated_facts', [])
            agent_utterance = turn.get('agent_utterance', '')
            user_utterance = turn.get('user_utterance', '')
            
            # Grounding metrics
            for fact_id in facts:
                total_facts_mentioned += 1
                unique_facts.add(fact_id)
                fact_mentions[fact_id] = fact_mentions.get(fact_id, 0) + 1
                kb_citations += 1
                true_positives += 1
            
            false_positives += len(hallucinated)
            hallucinations += len(hallucinated)
            
            # On-plan compliance checking
            if option == 'Explain':
                if subaction == 'ExplainNewFact':
                    # Should introduce NEW fact IDs
                    new_facts = [f for f in facts if fact_mentions.get(f, 0) == 1]
                    if new_facts:
                        on_plan_compliant['ExplainNewFact'] += 1
                    else:
                        on_plan_violations['ExplainNewFact'] += 1
                elif subaction in ['RepeatFact', 'ClarifyFact']:
                    # Should NOT introduce new fact IDs
                    new_facts = [f for f in facts if fact_mentions.get(f, 0) == 1]
                    if not new_facts:
                        on_plan_compliant[subaction] += 1
                    else:
                        on_plan_violations[subaction] += 1
            elif option == 'AskQuestion':
                # Should end with a question
                if agent_utterance.strip().endswith('?'):
                    on_plan_compliant['AskQuestion'] += 1
                else:
                    on_plan_violations['AskQuestion'] += 1
            
            # Act classifier agreement
            if agent_utterance:
                # Classify agent utterance (for realized act)
                realized_act = act_classifier.classify(agent_utterance)
                # Expected act based on option/subaction
                expected_act = 'question' if option == 'AskQuestion' else 'statement'
                agreement = (realized_act == expected_act)
                act_agreements.append(1 if agreement else 0)
            
            # Header completeness (approximate - based on available info)
            # Count filled slots: option, subaction, exhibit, facts
            completeness = 0.0
            max_slots = 4.0
            if option and option != 'Unknown':
                completeness += 1.0
            if subaction and subaction != 'Unknown':
                completeness += 1.0
            if turn.get('exhibit'):
                completeness += 1.0
            if facts:
                completeness += 1.0
            header_completeness_scores.append(completeness / max_slots)
            
            # Track completeness vs errors
            errors_this_turn = len(hallucinated) + (1 if (act_agreements and not act_agreements[-1]) else 0)
            completeness_vs_errors.append({
                'completeness': completeness / max_slots,
                'errors': errors_this_turn
            })
        
        # Compute metrics
        if total_facts_mentioned > 0:
            metrics['novel_fact_coverage'] = len(unique_facts) / total_facts_mentioned
            metrics['repetition_ratio'] = sum(1 for count in fact_mentions.values() if count > 1) / len(fact_mentions) if fact_mentions else 0.0
            metrics['kb_citation_rate'] = kb_citations / total_facts_mentioned
            metrics['hallucination_rate'] = hallucinations / total_facts_mentioned
        
        # Grounding precision/recall
        if true_positives + false_positives > 0:
            metrics['grounding_precision'] = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives > 0:
            # Recall is approximate (we don't track all valid facts not mentioned)
            metrics['grounding_recall'] = true_positives / (true_positives + false_negatives) if false_negatives > 0 else 1.0
        
        # On-plan compliance
        total_checks = {}
        for key in set(list(on_plan_compliant.keys()) + list(on_plan_violations.keys())):
            total = on_plan_compliant.get(key, 0) + on_plan_violations.get(key, 0)
            if total > 0:
                metrics['on_plan_compliance'][key] = {
                    'compliant': on_plan_compliant.get(key, 0),
                    'violations': on_plan_violations.get(key, 0),
                    'compliance_rate': on_plan_compliant.get(key, 0) / total
                }
        
        # Act classifier agreement
        if act_agreements:
            metrics['act_classifier_agreement'] = np.mean(act_agreements)
        
        # Header completeness
        if header_completeness_scores:
            metrics['header_completeness'] = float(np.mean(header_completeness_scores))
        
        # Correlation: header completeness vs errors
        if completeness_vs_errors:
            completeness_vals = [x['completeness'] for x in completeness_vs_errors]
            error_vals = [x['errors'] for x in completeness_vs_errors]
            if len(completeness_vals) > 1:
                try:
                    corr = np.corrcoef(completeness_vals, error_vals)[0, 1]
                    metrics['header_completeness_vs_errors'] = {
                        'correlation': float(corr) if not np.isnan(corr) else 0.0,
                        'expected_negative': corr < 0 if not np.isnan(corr) else None
                    }
                except:
                    metrics['header_completeness_vs_errors'] = None
        
        return metrics
    
    def compute_h7_metrics(self) -> Dict[str, Any]:
        """
        Compute H7-specific metrics (hybrid BERT for multi-turn context).
        
        Returns:
            Dictionary of H7 metrics:
            - dialogue_coherence: Reference resolution accuracy (pronouns/ellipsis)
            - context_dependent_qa_rate: Fraction of multi-turn questions answered correctly
            - response_appropriateness: Contextual vs generic classification score
            - contradiction_rate: Instances where agent contradicts earlier statements
            - embedding_similarity: Cosine similarity between consecutive c_t embeddings
            - reference_tracking_accuracy: Pronoun/ellipsis resolution accuracy
        """
        metrics = {
            'dialogue_coherence': 0.0,
            'context_dependent_qa_rate': 0.0,
            'response_appropriateness': 0.0,
            'contradiction_rate': 0.0,
            'embedding_similarity': [],
            'mean_embedding_similarity': 0.0,
            'reference_tracking_accuracy': 0.0,
        }
        
        # Load RL metrics if available (for embedding similarity)
        if self.experiment_path and self.experiment_path.exists():
            rl_metrics_files = list((self.experiment_path / 'logs').glob('rl_metrics_*.json'))
            if rl_metrics_files:
                try:
                    with open(rl_metrics_files[0], 'r') as f:
                        rl_metrics = json.load(f)
                        # Extract embedding similarity if available
                        if 'embedding_similarity' in rl_metrics:
                            metrics['embedding_similarity'] = rl_metrics['embedding_similarity']
                            if metrics['embedding_similarity']:
                                metrics['mean_embedding_similarity'] = float(np.mean(metrics['embedding_similarity']))
                except Exception as e:
                    pass  # RL metrics not available yet
        
        # Track dialogue coherence (reference resolution)
        reference_resolutions = []
        context_dependent_qa = []
        contradictions = []
        response_appropriateness_scores = []
        
        # Analyze turn data for multi-turn understanding
        for i, turn in enumerate(self.turn_data):
            agent_utterance = turn.get('agent_utterance', '')
            user_utterance = turn.get('user_utterance', '')
            
            # Check for pronouns/ellipsis that need resolution
            pronouns = ['it', 'that', 'this', 'they', 'them', 'these', 'those']
            has_pronoun = any(pronoun in user_utterance.lower().split() for pronoun in pronouns)
            
            if has_pronoun and i > 0:
                # Check if agent response addresses the reference
                # Simple heuristic: agent mentions keywords from previous turns
                prev_turns = self.turn_data[max(0, i-3):i]
                prev_keywords = set()
                for prev_turn in prev_turns:
                    prev_utt = prev_turn.get('agent_utterance', '') + ' ' + prev_turn.get('user_utterance', '')
                    # Extract keywords (simple: words > 4 chars, not common words)
                    common_words = {'the', 'and', 'that', 'this', 'with', 'from', 'about', 'what', 'when', 'where'}
                    keywords = [w.lower() for w in prev_utt.split() if len(w) > 4 and w.lower() not in common_words]
                    prev_keywords.update(keywords)
                
                # Check if agent response contains relevant keywords
                agent_keywords = set([w.lower() for w in agent_utterance.split() if len(w) > 4])
                overlap = len(agent_keywords & prev_keywords)
                resolution_score = min(1.0, overlap / max(1, len(prev_keywords) * 0.3))  # 30% overlap = good
                reference_resolutions.append(resolution_score)
            
            # Context-dependent question answering
            if '?' in user_utterance:
                # Check if question requires context from previous turns
                context_indicators = ['that', 'it', 'this', 'earlier', 'before', 'mentioned', 'said']
                needs_context = any(indicator in user_utterance.lower() for indicator in context_indicators)
                
                if needs_context:
                    # Check if agent response addresses the question with context
                    # Simple: agent response is longer and contains relevant keywords
                    if len(agent_utterance) > 50:  # Substantive response
                        context_dependent_qa.append(1.0)
                    else:
                        context_dependent_qa.append(0.0)
            
            # Response appropriateness (contextual vs generic)
            # Generic responses are short, don't reference previous context
            is_generic = len(agent_utterance) < 30 or agent_utterance.lower().startswith(('yes', 'no', 'ok', 'sure'))
            appropriateness = 0.0 if is_generic else 1.0
            response_appropriateness_scores.append(appropriateness)
            
            # Contradiction detection (simplified)
            # Check if agent contradicts facts mentioned earlier
            if i > 0:
                # Extract fact IDs from current and previous utterances
                import re
                current_facts = set(re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance))
                prev_facts = set()
                for prev_turn in self.turn_data[max(0, i-5):i]:
                    prev_utt = prev_turn.get('agent_utterance', '')
                    prev_facts.update(re.findall(r'\[([A-Z]{2}_\d{3})\]', prev_utt))
                
                # Simple contradiction: agent says opposite of previous statement
                # (This is simplified - full implementation would need semantic analysis)
                contradictions.append(0)  # Placeholder - would need more sophisticated analysis
        
        # Compute final metrics
        if reference_resolutions:
            metrics['dialogue_coherence'] = float(np.mean(reference_resolutions))
            metrics['reference_tracking_accuracy'] = float(np.mean(reference_resolutions))
        
        if context_dependent_qa:
            metrics['context_dependent_qa_rate'] = float(np.mean(context_dependent_qa))
        
        if response_appropriateness_scores:
            metrics['response_appropriateness'] = float(np.mean(response_appropriateness_scores))
        
        if contradictions:
            metrics['contradiction_rate'] = float(np.mean(contradictions))
        
        return metrics
    
    def save_metrics(self, hypothesis: str):
        """
        Save computed metrics to JSON file.
        
        Args:
            hypothesis: Hypothesis identifier ('h1', 'h2', 'h3', 'h5', 'h6', or 'h7')
        """
        all_metrics = {
            'experiment_name': self.experiment_name,
            'hypothesis': hypothesis,
            'common_metrics': self.compute_common_metrics(),
        }
        
        if hypothesis == 'h1':
            all_metrics['h1_metrics'] = self.compute_h1_metrics()
        elif hypothesis == 'h2':
            all_metrics['h2_metrics'] = self.compute_h2_metrics()
        elif hypothesis == 'h3':
            all_metrics['h3_metrics'] = self.compute_h3_metrics()
        elif hypothesis == 'h5':
            all_metrics['h5_metrics'] = self.compute_h5_metrics()
        elif hypothesis == 'h6':
            all_metrics['h6_metrics'] = self.compute_h6_metrics()
        elif hypothesis == 'h7':
            all_metrics['h7_metrics'] = self.compute_h7_metrics()
        
        output_file = self.results_dir / f'{self.experiment_name}_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Also save to major_results/ if applicable
        self._save_to_major_results(all_metrics, hypothesis)
        
        # Generate plots automatically
        self._generate_plots(hypothesis, all_metrics)
        
        return all_metrics
    
    def _save_to_major_results(self, metrics: Dict, hypothesis: str):
        """
        Save evaluation metrics to major_results/ directory.
        
        Args:
            metrics: Computed metrics dictionary
            hypothesis: Hypothesis identifier
        """
        try:
            from src.utils.major_results_manager import MajorResultsManager
            
            # Detect model name from experiment name
            manager = MajorResultsManager()
            model_name = manager.normalize_model_name(self.experiment_name)
            
            # Get latest experiment for this model (or create new one)
            exp_dir = manager.get_latest_experiment(model_name)
            if exp_dir is None:
                # No experiment exists yet, create new one
                exp_dir = manager.create_experiment_folder(model_name)
            
            # Save to evaluation/ directory
            eval_dir = exp_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output filename based on hypothesis
            if hypothesis in ['h1', 'h2', 'h3', 'h5', 'h6', 'h7']:
                output_filename = f"{hypothesis}_metrics.json"
            else:
                output_filename = f"{model_name}_metrics.json"
            
            output_file = eval_dir / output_filename
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved evaluation metrics to major_results: {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save to major_results: {e}")
    
    def _generate_plots(self, hypothesis: str, metrics: Dict):
        """
        Automatically generate plots for the hypothesis.
        
        Args:
            hypothesis: Hypothesis identifier
            metrics: Computed metrics dictionary
        """
        try:
            from experiments.shared.evaluation_visualizations import generate_hypothesis_plots
            from src.utils.major_results_manager import MajorResultsManager
            
            # Get experiment directory for saving plots
            manager = MajorResultsManager()
            model_name = manager.normalize_model_name(self.experiment_name)
            exp_dir = manager.get_latest_experiment(model_name)
            
            if exp_dir is None:
                logger.warning("Could not find experiment directory for plot generation")
                return
            
            # Create visualization directory
            viz_dir = exp_dir / "visualizations" / "evaluation"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Load baseline metrics if needed for comparison
            baseline_metrics = None
            comparison_metrics = None
            training_logs = None
            
            if hypothesis in ['h1', 'h3', 'h5', 'h6', 'h7']:
                # Need baseline for comparison
                baseline_exp = manager.get_latest_experiment('baseline')
                if baseline_exp:
                    # Try to load baseline metrics for this hypothesis
                    baseline_metrics_file = baseline_exp / "evaluation" / f"{hypothesis}_metrics.json"
                    if baseline_metrics_file.exists():
                        with open(baseline_metrics_file, 'r') as f:
                            baseline_metrics = json.load(f)
                    else:
                        # Try to compute baseline metrics from training data
                        try:
                            baseline_evaluator = HypothesisEvaluator('baseline', baseline_exp / 'evaluation')
                            baseline_evaluator.load_experiment_data(baseline_exp)
                            # Compute the same hypothesis metrics for baseline
                            if hypothesis == 'h1':
                                baseline_metrics = {
                                    'common_metrics': baseline_evaluator.compute_common_metrics(),
                                    'h1_metrics': baseline_evaluator.compute_h1_metrics()
                                }
                            elif hypothesis == 'h3':
                                baseline_metrics = {
                                    'common_metrics': baseline_evaluator.compute_common_metrics(),
                                    'h3_metrics': baseline_evaluator.compute_h3_metrics()
                                }
                            elif hypothesis == 'h5':
                                baseline_metrics = {
                                    'common_metrics': baseline_evaluator.compute_common_metrics(),
                                    'h5_metrics': baseline_evaluator.compute_h5_metrics()
                                }
                            elif hypothesis == 'h6':
                                baseline_metrics = {
                                    'common_metrics': baseline_evaluator.compute_common_metrics(),
                                    'h6_metrics': baseline_evaluator.compute_h6_metrics()
                                }
                            elif hypothesis == 'h7':
                                baseline_metrics = {
                                    'common_metrics': baseline_evaluator.compute_common_metrics(),
                                    'h7_metrics': baseline_evaluator.compute_h7_metrics()
                                }
                        except Exception as e:
                            logger.warning(f"Could not compute baseline metrics for {hypothesis}: {e}")
                            baseline_metrics = None
                
                # Current model is the comparison
                comparison_metrics = metrics
            
            elif hypothesis == 'h2':
                # H2 only needs its own metrics
                pass
            
            elif hypothesis == 'h4':
                # H4 needs baseline + H1 metrics and training logs
                baseline_exp = manager.get_latest_experiment('baseline')
                h1_exp = manager.get_latest_experiment('h1_flat_policy')
                
                if baseline_exp:
                    baseline_metrics_file = baseline_exp / "evaluation" / "h1_metrics.json"
                    if baseline_metrics_file.exists():
                        with open(baseline_metrics_file, 'r') as f:
                            baseline_metrics = json.load(f)
                
                if h1_exp:
                    h1_metrics_file = h1_exp / "evaluation" / "h1_metrics.json"
                    if h1_metrics_file.exists():
                        with open(h1_metrics_file, 'r') as f:
                            comparison_metrics = json.load(f)
                
                # Load training logs
                if baseline_exp and h1_exp:
                    training_logs = {
                        'baseline': self._load_training_logs(baseline_exp),
                        'h1': self._load_training_logs(h1_exp)
                    }
            
            # Generate plots
            generate_hypothesis_plots(
                hypothesis=hypothesis,
                metrics=metrics,
                output_dir=viz_dir,
                baseline_metrics=baseline_metrics,
                comparison_metrics=comparison_metrics,
                training_logs=training_logs
            )
            
            logger.info(f"Generated plots for {hypothesis} in {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots for {hypothesis}: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_training_logs(self, exp_dir: Path) -> Dict:
        """Load training logs from experiment directory."""
        try:
            logs = {}
            logs_dir = exp_dir / "training" / "logs"
            
            # Load learning curves
            learning_curves_files = list(logs_dir.glob("learning_curves_*.json"))
            if learning_curves_files:
                with open(learning_curves_files[0], 'r') as f:
                    learning_curves = json.load(f)
                    logs['episode_returns'] = learning_curves.get('episode_returns', [])
            
            # Load RL metrics for update norms
            rl_metrics_files = list(logs_dir.glob("rl_metrics_*.json"))
            if rl_metrics_files:
                with open(rl_metrics_files[0], 'r') as f:
                    rl_metrics = json.load(f)
                    logs['update_norms'] = rl_metrics.get('update_norms', [])
            
            return logs
        except Exception as e:
            logger.warning(f"Failed to load training logs: {e}")
            return {}

