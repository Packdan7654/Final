"""
Metrics Validator

Ensures all required metrics are saved for each model variation.
Provides validation and warnings for missing metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsValidator:
    """
    Validates that all required metrics are saved for a model.
    
    Checks for:
    - Training metrics (metrics_tracker, rl_metrics, learning_curves, convergence)
    - Evaluation metrics (hypothesis-specific metrics)
    - Visualizations (basic plots)
    - Checkpoints (for recovery)
    """
    
    REQUIRED_TRAINING_METRICS = [
        'training_metrics.json',
        'rl_metrics.json',
        'learning_curves.json',
        'convergence_report.json'
    ]
    
    REQUIRED_TRAINING_FILES = [
        'logs/metrics_tracker_*.json',
        'logs/rl_metrics_*.json',
        'logs/learning_curves_*.json',
        'logs/convergence_report_*.json',
        'metadata.json'
    ]
    
    REQUIRED_VISUALIZATIONS = [
        'visualizations/basic/learning_curve.png',
        'visualizations/basic/convergence_analysis.png',
        'visualizations/basic/rl_metrics_summary.png',  # Added: RL metrics summary plot
    ]
    
    def __init__(self, base_dir: str = "major_results"):
        """
        Initialize MetricsValidator.
        
        Args:
            base_dir: Base directory for major_results
        """
        self.base_dir = Path(base_dir)
    
    def validate_training_metrics(self, exp_dir: Path) -> Dict[str, bool]:
        """
        Validate training metrics exist.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary mapping metric names to existence status
        """
        metrics_dir = exp_dir / "metrics"
        training_dir = exp_dir / "training"
        
        status = {}
        
        # Check consolidated metrics
        for metric_file in self.REQUIRED_TRAINING_METRICS:
            metric_path = metrics_dir / metric_file
            status[f"metrics/{metric_file}"] = metric_path.exists()
        
        # Check training logs (source files)
        logs_dir = training_dir / "logs"
        if logs_dir.exists():
            for pattern in self.REQUIRED_TRAINING_FILES:
                if pattern.endswith('*.json'):
                    # Check for any matching file
                    base_pattern = pattern.replace('logs/', '').replace('*.json', '')
                    matches = list(logs_dir.glob(f"{base_pattern}*.json"))
                    status[f"training/{pattern}"] = len(matches) > 0
                else:
                    file_path = training_dir / pattern
                    status[f"training/{pattern}"] = file_path.exists()
        else:
            for pattern in self.REQUIRED_TRAINING_FILES:
                status[f"training/{pattern}"] = False
        
        return status
    
    def validate_evaluation_metrics(self, exp_dir: Path) -> Dict[str, bool]:
        """
        Validate evaluation metrics exist.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary mapping metric names to existence status
        """
        eval_dir = exp_dir / "evaluation"
        
        status = {}
        
        # Check for evaluation JSON files
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.json"))
            status['evaluation_metrics_exist'] = len(eval_files) > 0
            
            # Check for specific model metrics (extract model name from exp_dir)
            # Pattern: YYYYMMDD_NNN_modelname -> extract modelname
            import re
            match = re.match(r'^\d{8}_\d{3}_(.+)$', exp_dir.name)
            if match:
                model_name = match.group(1)
                expected_metrics = {
                    'baseline': 'baseline_metrics.json',
                    'h1_flat_policy': 'h1_metrics.json',
                    'h2_learned_terminations': 'h2_metrics.json',
                    'h3_minimal_prompts': 'h3_metrics.json',
                    'h5_state_ablation': 'h5_metrics.json',
                    'h6_transition_reward': 'h6_metrics.json',
                    'h7_hybrid_bert': 'h7_metrics.json',
                }
                
                expected_file = expected_metrics.get(model_name)
                if expected_file:
                    status[f'evaluation/{expected_file}'] = (eval_dir / expected_file).exists()
        else:
            status['evaluation_metrics_exist'] = False
        
        return status
    
    def validate_visualizations(self, exp_dir: Path) -> Dict[str, bool]:
        """
        Validate visualizations exist.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary mapping visualization names to existence status
        """
        viz_dir = exp_dir / "visualizations"
        
        status = {}
        
        for viz_file in self.REQUIRED_VISUALIZATIONS:
            viz_path = exp_dir / viz_file
            status[viz_file] = viz_path.exists()
        
        # Check for any visualizations in basic/
        basic_dir = viz_dir / "basic"
        if basic_dir.exists():
            basic_plots = list(basic_dir.glob("*.png")) + list(basic_dir.glob("*.pdf"))
            status['basic_visualizations_count'] = len(basic_plots)
        else:
            status['basic_visualizations_count'] = 0
        
        return status
    
    def validate_checkpoints(self, exp_dir: Path) -> Dict[str, bool]:
        """
        Validate checkpoints exist for recovery.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary mapping checkpoint info to status
        """
        checkpoint_dir = exp_dir / "training" / "checkpoints"
        
        status = {}
        
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
            status['checkpoints_exist'] = len(checkpoint_files) > 0
            status['checkpoint_count'] = len(checkpoint_files)
            
            # Check for latest checkpoint
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                status['latest_checkpoint'] = latest.name
        else:
            status['checkpoints_exist'] = False
            status['checkpoint_count'] = 0
        
        return status
    
    def validate_all(self, exp_dir: Path) -> Dict[str, Dict[str, bool]]:
        """
        Validate all metrics for an experiment.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary with validation results for each category
        """
        return {
            'training_metrics': self.validate_training_metrics(exp_dir),
            'evaluation_metrics': self.validate_evaluation_metrics(exp_dir),
            'visualizations': self.validate_visualizations(exp_dir),
            'checkpoints': self.validate_checkpoints(exp_dir)
        }
    
    def generate_validation_report(self, exp_dir: Path) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Validation report as string
        """
        results = self.validate_all(exp_dir)
        
        report_lines = [f"Validation Report for: {exp_dir.name}", "=" * 60, ""]
        
        # Training metrics
        report_lines.append("TRAINING METRICS:")
        training = results['training_metrics']
        all_training_ok = all(training.values())
        for metric, exists in training.items():
            status = "✓" if exists else "✗"
            report_lines.append(f"  {status} {metric}")
        report_lines.append(f"Status: {'✓ All present' if all_training_ok else '✗ Missing items'}")
        report_lines.append("")
        
        # Evaluation metrics
        report_lines.append("EVALUATION METRICS:")
        evaluation = results['evaluation_metrics']
        all_eval_ok = all(evaluation.values())
        for metric, exists in evaluation.items():
            status = "✓" if exists else "✗"
            report_lines.append(f"  {status} {metric}")
        report_lines.append(f"Status: {'✓ All present' if all_eval_ok else '✗ Missing items'}")
        report_lines.append("")
        
        # Visualizations
        report_lines.append("VISUALIZATIONS:")
        visualizations = results['visualizations']
        viz_count = visualizations.get('basic_visualizations_count', 0)
        report_lines.append(f"  Basic visualizations: {viz_count}")
        for viz, exists in visualizations.items():
            if viz != 'basic_visualizations_count':
                status = "✓" if exists else "✗"
                report_lines.append(f"  {status} {viz}")
        report_lines.append("")
        
        # Checkpoints
        report_lines.append("CHECKPOINTS:")
        checkpoints = results['checkpoints']
        if checkpoints.get('checkpoints_exist'):
            report_lines.append(f"  ✓ Checkpoints available: {checkpoints.get('checkpoint_count', 0)}")
            if 'latest_checkpoint' in checkpoints:
                report_lines.append(f"  Latest: {checkpoints['latest_checkpoint']}")
        else:
            report_lines.append("  ✗ No checkpoints found")
        report_lines.append("")
        
        # Overall status
        all_ok = all_training_ok and all_eval_ok and viz_count > 0
        report_lines.append("=" * 60)
        report_lines.append(f"OVERALL STATUS: {'✓ COMPLETE' if all_ok else '⚠ INCOMPLETE'}")
        
        return "\n".join(report_lines)
    
    def save_validation_report(self, exp_dir: Path, output_path: Optional[Path] = None):
        """
        Save validation report to file.
        
        Args:
            exp_dir: Experiment directory in major_results/
            output_path: Optional output path (defaults to exp_dir/validation_report.txt)
        """
        report = self.generate_validation_report(exp_dir)
        
        if output_path is None:
            output_path = exp_dir / "validation_report.txt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to: {output_path}")
        return output_path

