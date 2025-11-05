"""
Episode-level file logger
Saves verbose turn details to episode-specific files
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager

class EpisodeLogger:
    """Manages episode-level logging to files"""
    
    def __init__(self, experiment_dir, enabled=True):
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.enabled = enabled and experiment_dir is not None
        self.episode_dir = None
        self.log_file = None
        
        if self.enabled:
            self.details_dir = Path(experiment_dir) / "episode_details"
            self.details_dir.mkdir(exist_ok=True)
    
    def start_episode(self, episode_num):
        """Start logging for a new episode"""
        if not self.enabled:
            return
        
        self.episode_dir = self.details_dir / f"episode_{episode_num:03d}"
        self.episode_dir.mkdir(exist_ok=True)
        
        log_path = self.episode_dir / "episode_log.txt"
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def end_episode(self):
        """End episode logging"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    @contextmanager
    def redirect_output(self):
        """Context manager to redirect stdout to episode log file"""
        if not self.enabled or not self.log_file:
            yield
            return
        
        # Save original stdout
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # Redirect to log file
            sys.stdout = self.log_file
            sys.stderr = self.log_file
            yield
        finally:
            # Restore original
            sys.stdout = original_stdout
            sys.stderr = original_stderr

