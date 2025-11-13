"""
Comparison Tools for Hypothesis Testing

Provides tools to compare results across experimental variants
(baseline vs H5, baseline vs H6, etc.)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class HypothesisComparator:
    """
    Compare results across experimental variants for hypothesis testing.
    """
    
    def __init__(self, results_base_dir: Path):
        """
        Initialize comparator.
        
        Args:
            results_base_dir: Base directory containing all experiment results
        """
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(parents=True, exist_ok=True)
    
    def load_variant_metrics(self, variant_name: str) -> Optional[Dict]:
        """
        Load metrics for a specific variant.
        
        Args:
            variant_name: Name of variant (baseline, h5, h6)
            
        Returns:
            Metrics dictionary or None if not found
        """
        metrics_file = self.results_base_dir / f'{variant_name}_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def compare_h5_vs_baseline(self) -> Dict[str, Any]:
        """
        Compare H5 (state ablation) vs baseline.
        
        Returns:
            Comparison results dictionary
        """
        baseline_metrics = self.load_variant_metrics('baseline')
        h5_metrics = self.load_variant_metrics('h5')
        
        if not baseline_metrics or not h5_metrics:
            return {'error': 'Missing metrics files'}
        
        comparison = {
            'hypothesis': 'H5',
            'state_dimension': {
                'baseline': baseline_metrics.get('common_metrics', {}).get('state_dimension', 149),
                'h5': h5_metrics.get('h5_metrics', {}).get('state_dimension', None),
                'reduction': None,
            },
            'performance': {
                'baseline': {
                    'mean_return': baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': baseline_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
                'h5': {
                    'mean_return': h5_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': h5_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
            },
            'statistical_test': {
                'return_difference': None,
                'coverage_difference': None,
            },
        }
        
        # Compute differences
        baseline_return = comparison['performance']['baseline']['mean_return']
        h5_return = comparison['performance']['h5']['mean_return']
        comparison['statistical_test']['return_difference'] = h5_return - baseline_return
        
        baseline_coverage = comparison['performance']['baseline']['mean_coverage']
        h5_coverage = comparison['performance']['h5']['mean_coverage']
        comparison['statistical_test']['coverage_difference'] = h5_coverage - baseline_coverage
        
        # State dimension reduction
        baseline_dim = comparison['state_dimension']['baseline']
        h5_dim = comparison['state_dimension']['h5']
        if h5_dim:
            comparison['state_dimension']['reduction'] = baseline_dim - h5_dim
            comparison['state_dimension']['compression_ratio'] = (
                1.0 - (h5_dim / baseline_dim)
            )
        
        return comparison
    
    def compare_h6_vs_baseline(self) -> Dict[str, Any]:
        """
        Compare H6 (transition reward) vs baseline.
        
        Returns:
            Comparison results dictionary
        """
        baseline_metrics = self.load_variant_metrics('baseline')
        h6_metrics = self.load_variant_metrics('h6')
        
        if not baseline_metrics or not h6_metrics:
            return {'error': 'Missing metrics files'}
        
        comparison = {
            'hypothesis': 'H6',
            'transition_metrics': {
                'baseline': {
                    'success_rate': baseline_metrics.get('h6_metrics', {}).get('transition_success_rate', 0.0),
                    'avg_facts_before': baseline_metrics.get('h6_metrics', {}).get('avg_facts_before_transition', 0.0),
                },
                'h6': {
                    'success_rate': h6_metrics.get('h6_metrics', {}).get('transition_success_rate', 0.0),
                    'avg_facts_before': h6_metrics.get('h6_metrics', {}).get('avg_facts_before_transition', 0.0),
                },
            },
            'performance': {
                'baseline': {
                    'mean_return': baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': baseline_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
                'h6': {
                    'mean_return': h6_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': h6_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
            },
            'statistical_test': {
                'transition_success_difference': None,
                'return_difference': None,
            },
        }
        
        # Compute differences
        baseline_success = comparison['transition_metrics']['baseline']['success_rate']
        h6_success = comparison['transition_metrics']['h6']['success_rate']
        comparison['statistical_test']['transition_success_difference'] = h6_success - baseline_success
        
        baseline_return = comparison['performance']['baseline']['mean_return']
        h6_return = comparison['performance']['h6']['mean_return']
        comparison['statistical_test']['return_difference'] = h6_return - baseline_return
        
        return comparison
    
    def compare_h1_vs_baseline(self) -> Dict[str, Any]:
        """Compare H1 (flat policy) vs baseline (hierarchical)."""
        baseline_metrics = self.load_variant_metrics('baseline')
        h1_metrics = self.load_variant_metrics('h1')
        
        if not baseline_metrics or not h1_metrics:
            return {'error': 'Missing metrics files'}
        
        comparison = {
            'hypothesis': 'H1',
            'performance': {
                'baseline': {
                    'mean_return': baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': baseline_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
                'h1': {
                    'mean_return': h1_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                    'mean_coverage': h1_metrics.get('common_metrics', {}).get('mean_coverage', 0.0),
                },
            },
            'option_structure': {
                'baseline': {
                    'mean_coherent_span': baseline_metrics.get('h1_metrics', {}).get('mean_coherent_span', 0.0),
                    'switch_rate': baseline_metrics.get('h1_metrics', {}).get('switch_rate_per_100_turns', 0.0),
                },
                'h1': {
                    'mean_coherent_span': h1_metrics.get('h1_metrics', {}).get('mean_coherent_span', 0.0),
                    'switch_rate': h1_metrics.get('h1_metrics', {}).get('switch_rate_per_100_turns', 0.0),
                },
            },
        }
        
        return comparison
    
    def compare_h2_vs_baseline(self) -> Dict[str, Any]:
        """Compare H2 (fixed duration) vs baseline (learned terminations)."""
        baseline_metrics = self.load_variant_metrics('baseline')
        h2_metrics = self.load_variant_metrics('h2')
        
        if not baseline_metrics or not h2_metrics:
            return {'error': 'Missing metrics files'}
        
        comparison = {
            'hypothesis': 'H2',
            'termination_metrics': {
                'baseline': {
                    'mean_explain_duration': baseline_metrics.get('h2_metrics', {}).get('mean_explain_duration', 0.0),
                    'dwell_correlation': baseline_metrics.get('h2_metrics', {}).get('dwell_correlation', 0.0),
                },
                'h2': {
                    'mean_explain_duration': h2_metrics.get('h2_metrics', {}).get('mean_explain_duration', 0.0),
                    'dwell_correlation': h2_metrics.get('h2_metrics', {}).get('dwell_correlation', 0.0),
                },
            },
            'performance': {
                'baseline': {
                    'mean_return': baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                },
                'h2': {
                    'mean_return': h2_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                },
            },
        }
        
        return comparison
    
    def compare_h3_vs_baseline(self) -> Dict[str, Any]:
        """Compare H3 (minimal prompts) vs baseline (structured headers)."""
        baseline_metrics = self.load_variant_metrics('baseline')
        h3_metrics = self.load_variant_metrics('h3')
        
        if not baseline_metrics or not h3_metrics:
            return {'error': 'Missing metrics files'}
        
        comparison = {
            'hypothesis': 'H3',
            'prompt_metrics': {
                'baseline': {
                    'novel_fact_coverage': baseline_metrics.get('h3_metrics', {}).get('novel_fact_coverage', 0.0),
                    'repetition_ratio': baseline_metrics.get('h3_metrics', {}).get('repetition_ratio', 0.0),
                    'hallucination_rate': baseline_metrics.get('h3_metrics', {}).get('hallucination_rate', 0.0),
                },
                'h3': {
                    'novel_fact_coverage': h3_metrics.get('h3_metrics', {}).get('novel_fact_coverage', 0.0),
                    'repetition_ratio': h3_metrics.get('h3_metrics', {}).get('repetition_ratio', 0.0),
                    'hallucination_rate': h3_metrics.get('h3_metrics', {}).get('hallucination_rate', 0.0),
                },
            },
            'performance': {
                'baseline': {
                    'mean_return': baseline_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                },
                'h3': {
                    'mean_return': h3_metrics.get('common_metrics', {}).get('mean_return', 0.0),
                },
            },
        }
        
        return comparison
    
    def generate_comparison_report(self, output_file: Path):
        """
        Generate comprehensive comparison report for all hypotheses.
        
        Args:
            output_file: Path to save comparison report
        """
        h1_comparison = self.compare_h1_vs_baseline()
        h2_comparison = self.compare_h2_vs_baseline()
        h3_comparison = self.compare_h3_vs_baseline()
        h5_comparison = self.compare_h5_vs_baseline()
        h6_comparison = self.compare_h6_vs_baseline()
        
        report = {
            'h1_comparison': h1_comparison,
            'h2_comparison': h2_comparison,
            'h3_comparison': h3_comparison,
            'h5_comparison': h5_comparison,
            'h6_comparison': h6_comparison,
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate text summary
        summary_file = output_file.with_suffix('.txt')
        with open(summary_file, 'w') as f:
            f.write("HYPOTHESIS TESTING COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # H1
            f.write("H1: Option Structure vs Flat Policy\n")
            f.write("-" * 80 + "\n")
            if 'error' not in h1_comparison:
                perf = h1_comparison['performance']
                f.write(f"Return: Baseline={perf['baseline']['mean_return']:.3f}, H1={perf['h1']['mean_return']:.3f}\n")
                f.write(f"Difference: {perf['h1']['mean_return'] - perf['baseline']['mean_return']:.3f}\n")
            f.write("\n")
            
            # H2
            f.write("H2: Learned Terminations\n")
            f.write("-" * 80 + "\n")
            if 'error' not in h2_comparison:
                term = h2_comparison['termination_metrics']
                f.write(f"Dwell Correlation: Baseline={term['baseline']['dwell_correlation']:.3f}, H2={term['h2']['dwell_correlation']:.3f}\n")
            f.write("\n")
            
            # H3
            f.write("H3: Prompt Headers\n")
            f.write("-" * 80 + "\n")
            if 'error' not in h3_comparison:
                prompt = h3_comparison['prompt_metrics']
                f.write(f"Hallucination Rate: Baseline={prompt['baseline']['hallucination_rate']:.3f}, H3={prompt['h3']['hallucination_rate']:.3f}\n")
            f.write("\n")
            
            # H5
            f.write("H5: State Ablation\n")
            f.write("-" * 80 + "\n")
            if 'error' not in h5_comparison:
                f.write(f"State Dimension Reduction: {h5_comparison['state_dimension'].get('reduction', 'N/A')}\n")
                f.write(f"Return Difference: {h5_comparison['statistical_test']['return_difference']:.3f}\n")
            f.write("\n")
            
            # H6
            f.write("H6: Transition Reward Shaping\n")
            f.write("-" * 80 + "\n")
            if 'error' not in h6_comparison:
                f.write(f"Transition Success Difference: {h6_comparison['statistical_test']['transition_success_difference']:.3f}\n")
                f.write(f"Return Difference: {h6_comparison['statistical_test']['return_difference']:.3f}\n")
            f.write("\n")
        
        return report
    
    def perform_paired_statistical_test(self, baseline_values: List[float], 
                                        variant_values: List[float],
                                        test_type: str = 'wilcoxon') -> Dict[str, Any]:
        """
        Perform paired statistical test between baseline and variant.
        
        Args:
            baseline_values: List of values from baseline (per seed/episode)
            variant_values: List of values from variant (per seed/episode)
            test_type: 'wilcoxon' (default) or 'ttest'
        
        Returns:
            Dictionary with test results (statistic, p-value, effect_size)
        """
        if len(baseline_values) != len(variant_values):
            return {'error': 'Mismatched sample sizes'}
        
        if len(baseline_values) < 2:
            return {'error': 'Insufficient samples'}
        
        baseline_arr = np.array(baseline_values)
        variant_arr = np.array(variant_values)
        differences = variant_arr - baseline_arr
        
        results = {
            'test_type': test_type,
            'n_samples': len(baseline_values),
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences)),
        }
        
        if test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric, robust to outliers)
            try:
                statistic, p_value = stats.wilcoxon(baseline_arr, variant_arr, alternative='two-sided')
                results['statistic'] = float(statistic)
                results['p_value'] = float(p_value)
                results['significant'] = p_value < 0.05
            except:
                results['error'] = 'Wilcoxon test failed'
        elif test_type == 'ttest':
            # Paired t-test (parametric, assumes normality)
            try:
                statistic, p_value = stats.ttest_rel(baseline_arr, variant_arr)
                results['statistic'] = float(statistic)
                results['p_value'] = float(p_value)
                results['significant'] = p_value < 0.05
            except:
                results['error'] = 'T-test failed'
        
        # Effect size (Cohen's d for paired samples)
        if 'error' not in results and len(differences) > 1:
            pooled_std = np.sqrt((np.var(baseline_arr) + np.var(variant_arr)) / 2)
            if pooled_std > 0:
                cohens_d = np.mean(differences) / pooled_std
                results['effect_size'] = float(cohens_d)
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    results['effect_size_interpretation'] = 'negligible'
                elif abs(cohens_d) < 0.5:
                    results['effect_size_interpretation'] = 'small'
                elif abs(cohens_d) < 0.8:
                    results['effect_size_interpretation'] = 'medium'
                else:
                    results['effect_size_interpretation'] = 'large'
        
        return results

