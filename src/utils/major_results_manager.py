"""
Major Results Manager

Manages the organization of all training results, evaluations, and visualizations
in the major_results/ directory structure. Provides automatic organization,
metrics validation, and visualization generation.

New structure: Each experiment gets a folder named YYYYMMDD_NNN_modelname
where NNN is the experiment number (increments for each run of that model type).
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


class MajorResultsManager:
    """
    Manages major_results/ directory structure for organized experiment results.
    
    New structure: Each experiment gets its own dated, numbered folder:
    - YYYYMMDD_001_baseline/
    - YYYYMMDD_002_baseline/
    - YYYYMMDD_001_h1_flat_policy/
    - etc.
    
    Each folder contains:
    - README.md: Model description and configuration
    - training/: Training results (logs, checkpoints, models)
    - evaluation/: Evaluation metrics and results
    - visualizations/: Basic and advanced plots
    - metrics/: Consolidated metrics files
    """
    
    # Model name mappings
    MODEL_NAMES = {
        'baseline': 'baseline',
        'h1': 'h1_flat_policy',
        'h1_flat_policy': 'h1_flat_policy',
        'h2': 'h2_learned_terminations',
        'h2_learned_terminations': 'h2_learned_terminations',
        'h2_fixed_duration': 'h2_learned_terminations',
        'h3': 'h3_minimal_prompts',
        'h3_minimal_prompts': 'h3_minimal_prompts',
        'h5': 'h5_state_ablation',
        'h5_state_ablation': 'h5_state_ablation',
        'h6': 'h6_transition_reward',
        'h6_transition_reward': 'h6_transition_reward',
        'h7': 'h7_hybrid_bert',
        'h7_hybrid_bert': 'h7_hybrid_bert',
    }
    
    def __init__(self, base_dir: str = "major_results"):
        """
        Initialize MajorResultsManager.
        
        Args:
            base_dir: Base directory for major_results (default: "major_results")
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name to standard format.
        
        Args:
            model_name: Model name (e.g., 'baseline', 'h1', 'h1_flat_policy')
            
        Returns:
            Normalized model name (e.g., 'baseline', 'h1_flat_policy')
        """
        model_name_lower = model_name.lower().strip()
        return self.MODEL_NAMES.get(model_name_lower, model_name_lower)
    
    def get_next_experiment_number(self, model_name: str, date_str: str = None) -> int:
        """
        Get the next experiment number for a model type on a given date.
        
        Args:
            model_name: Normalized model name
            date_str: Date string in YYYYMMDD format (default: today)
            
        Returns:
            Next experiment number (1, 2, 3, ...)
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        # Find all existing experiments for this model on this date
        pattern = re.compile(rf"^{date_str}_(\d+)_({re.escape(model_name)})$")
        existing_numbers = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    existing_numbers.append(int(match.group(1)))
        
        if existing_numbers:
            return max(existing_numbers) + 1
        else:
            return 1
    
    def create_experiment_folder(self, model_name: str, description: str = None, 
                                 date_str: str = None, exp_num: int = None) -> Path:
        """
        Create folder structure for a new experiment run.
        
        Args:
            model_name: Model name (will be normalized)
            description: Optional description for README
            date_str: Date string in YYYYMMDD format (default: today)
            exp_num: Experiment number (default: auto-increment)
            
        Returns:
            Path to experiment folder
        """
        normalized_name = self.normalize_model_name(model_name)
        
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        if exp_num is None:
            exp_num = self.get_next_experiment_number(normalized_name, date_str)
        
        # Create folder name: YYYYMMDD_NNN_modelname
        folder_name = f"{date_str}_{exp_num:03d}_{normalized_name}"
        exp_dir = self.base_dir / folder_name
        
        # Create directory structure
        (exp_dir / "training" / "logs").mkdir(parents=True, exist_ok=True)
        (exp_dir / "training" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_dir / "training" / "models").mkdir(parents=True, exist_ok=True)
        (exp_dir / "training" / "maps").mkdir(parents=True, exist_ok=True)
        (exp_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        (exp_dir / "visualizations" / "basic").mkdir(parents=True, exist_ok=True)
        (exp_dir / "visualizations" / "advanced").mkdir(parents=True, exist_ok=True)
        (exp_dir / "visualizations" / "evaluation").mkdir(parents=True, exist_ok=True)
        (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created experiment folder: {exp_dir}")
        
        return exp_dir
    
    def create_model_folder(self, model_name: str, description: str = None) -> Path:
        """
        DEPRECATED: Use create_experiment_folder instead.
        Creates a new experiment folder with date and number.
        """
        return self.create_experiment_folder(model_name, description)
    
    def get_model_path(self, model_name: str) -> Path:
        """
        DEPRECATED: Get path to latest experiment for a model, or create new one.
        Use get_latest_experiment() or create_experiment_folder() instead.
        
        Args:
            model_name: Model name (will be normalized)
            
        Returns:
            Path to latest experiment folder (creates new if needed)
        """
        normalized_name = self.normalize_model_name(model_name)
        latest = self.get_latest_experiment(normalized_name)
        if latest:
            return latest
        return self.create_experiment_folder(normalized_name)
    
    def get_latest_experiment(self, model_name: str) -> Optional[Path]:
        """
        Get the latest experiment folder for a model type.
        
        Args:
            model_name: Model name (will be normalized)
            
        Returns:
            Path to latest experiment folder, or None if none exist
        """
        normalized_name = self.normalize_model_name(model_name)
        
        # Find all experiments for this model
        pattern = re.compile(rf"^\d{{8}}_\d{{3}}_{re.escape(normalized_name)}$")
        experiments = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                experiments.append(item)
        
        if not experiments:
            return None
        
        # Sort by name (which includes date and number) and return latest
        experiments.sort(key=lambda x: x.name, reverse=True)
        return experiments[0]
    
    def copy_training_results(self, source_dir: str, exp_dir: Path, 
                            copy_maps: bool = True) -> bool:
        """
        Copy training results from training_logs/ to major_results/.
        
        Args:
            source_dir: Source directory (from training_logs/experiments/...)
            model_name: Model name (will be normalized)
            copy_maps: Whether to copy map visualizations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.warning(f"Source directory does not exist: {source_path}")
                return False
            
            training_dir = exp_dir / "training"
            
            # Copy logs
            source_logs = source_path / "logs"
            if source_logs.exists():
                dest_logs = training_dir / "logs"
                self._copy_directory(source_logs, dest_logs)
                logger.info(f"Copied logs from {source_logs} to {dest_logs}")
            
            # Copy checkpoints
            source_checkpoints = source_path / "checkpoints"
            if source_checkpoints.exists():
                dest_checkpoints = training_dir / "checkpoints"
                self._copy_directory(source_checkpoints, dest_checkpoints)
                logger.info(f"Copied checkpoints from {source_checkpoints} to {dest_checkpoints}")
            
            # Copy models
            source_models = source_path / "models"
            if source_models.exists():
                dest_models = training_dir / "models"
                self._copy_directory(source_models, dest_models)
                logger.info(f"Copied models from {source_models} to {dest_models}")
            
            # Copy maps (if requested)
            if copy_maps:
                source_maps = source_path / "maps"
                if source_maps.exists():
                    dest_maps = training_dir / "maps"
                    self._copy_directory(source_maps, dest_maps)
                    logger.info(f"Copied maps from {source_maps} to {dest_maps}")
            
            # Copy metadata.json if exists
            source_metadata = source_path / "metadata.json"
            if source_metadata.exists():
                dest_metadata = training_dir / "metadata.json"
                shutil.copy2(source_metadata, dest_metadata)
                logger.info(f"Copied metadata from {source_metadata} to {dest_metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying training results: {e}")
            return False
    
    def copy_evaluation_results(self, source_dir: str, exp_dir: Path) -> bool:
        """
        Copy evaluation results from experiments/results/ to major_results/.
        
        Args:
            source_dir: Source directory (from experiments/results/{model}/)
            exp_dir: Experiment directory in major_results/
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.warning(f"Source directory does not exist: {source_path}")
                return False
            
            eval_dir = exp_dir / "evaluation"
            
            # Copy all JSON files
            for json_file in source_path.glob("*.json"):
                dest_file = eval_dir / json_file.name
                shutil.copy2(json_file, dest_file)
                logger.info(f"Copied evaluation file: {json_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying evaluation results: {e}")
            return False
    
    def consolidate_metrics(self, exp_dir: Path, training_dir: Path) -> bool:
        """
        Consolidate metrics from training logs into metrics/ folder.
        
        Args:
            exp_dir: Experiment directory in major_results/
            training_dir: Path to training directory (with logs/)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metrics_dir = exp_dir / "metrics"
            
            logs_dir = training_dir / "logs"
            if not logs_dir.exists():
                logger.warning(f"Logs directory does not exist: {logs_dir}")
                return False
            
            # Copy key metrics files
            metrics_files = {
                'metrics_tracker': 'training_metrics.json',
                'rl_metrics': 'rl_metrics.json',
                'learning_curves': 'learning_curves.json',
                'convergence_report': 'convergence_report.json'
            }
            
            for pattern, dest_name in metrics_files.items():
                source_files = list(logs_dir.glob(f"{pattern}_*.json"))
                if source_files:
                    # Use most recent file
                    latest_file = max(source_files, key=lambda p: p.stat().st_mtime)
                    dest_file = metrics_dir / dest_name
                    shutil.copy2(latest_file, dest_file)
                    logger.info(f"Consolidated {pattern} -> {dest_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error consolidating metrics: {e}")
            return False
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Get path to model folder in major_results/.
        
        Args:
            model_name: Model name (will be normalized)
            
        Returns:
            Path to model folder
        """
        normalized_name = self.normalize_model_name(model_name)
        return self.base_dir / normalized_name
    
    def ensure_metrics_saved(self, exp_dir: Path) -> Dict[str, bool]:
        """
        Verify all required metrics are saved.
        
        Args:
            exp_dir: Experiment directory in major_results/
            
        Returns:
            Dictionary mapping metric names to existence status
        """
        
        required_metrics = {
            'training_metrics': exp_dir / "metrics" / "training_metrics.json",
            'rl_metrics': exp_dir / "metrics" / "rl_metrics.json",
            'learning_curves': exp_dir / "metrics" / "learning_curves.json",
            'convergence_report': exp_dir / "metrics" / "convergence_report.json",
        }
        
        status = {}
        for name, path in required_metrics.items():
            status[name] = path.exists()
            if not path.exists():
                logger.warning(f"Missing metric: {name} at {path}")
        
        return status
    
    def _copy_directory(self, source: Path, dest: Path):
        """
        Copy directory contents, preserving structure.
        
        Args:
            source: Source directory
            dest: Destination directory
        """
        if not source.exists():
            return
        
        dest.mkdir(parents=True, exist_ok=True)
        
        for item in source.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                self._copy_directory(item, dest_item)
            else:
                shutil.copy2(item, dest_item)
    
    def list_all_experiments(self) -> List[Path]:
        """
        List all experiment folders in major_results/.
        
        Returns:
            List of experiment directory paths, sorted by date and number
        """
        experiments = []
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it matches the pattern YYYYMMDD_NNN_modelname
                    if re.match(r'^\d{8}_\d{3}_\w+$', item.name):
                        experiments.append(item)
        return sorted(experiments, key=lambda x: x.name, reverse=True)
    
    def list_experiments_by_model(self, model_name: str) -> List[Path]:
        """
        List all experiments for a specific model type.
        
        Args:
            model_name: Model name (will be normalized)
            
        Returns:
            List of experiment directory paths, sorted by date and number
        """
        normalized_name = self.normalize_model_name(model_name)
        pattern = re.compile(rf"^\d{{8}}_\d{{3}}_{re.escape(normalized_name)}$")
        
        experiments = []
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and pattern.match(item.name):
                    experiments.append(item)
        
        return sorted(experiments, key=lambda x: x.name, reverse=True)
    
    def list_all_models(self) -> List[str]:
        """
        DEPRECATED: List unique model types from all experiments.
        
        Returns:
            List of unique model names
        """
        models = set()
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Extract model name from YYYYMMDD_NNN_modelname
                    match = re.match(r'^\d{8}_\d{3}_(.+)$', item.name)
                    if match:
                        models.add(match.group(1))
        return sorted(models)

