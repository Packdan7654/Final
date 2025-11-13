"""
H2 Environment Variant: Fixed-Duration Options

Extends base environment to use fixed-duration options instead of
learned terminations for hypothesis testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.env import MuseumDialogueEnv


class H2FixedDurationEnv(MuseumDialogueEnv):
    """
    Environment variant for H2: fixed-duration options vs learned terminations.
    
    Uses fixed durations for Explain option instead of learned termination functions.
    Tests if learned terminations adapt better to engagement and intent.
    """
    
    def __init__(self, *args, fixed_explain_duration: int = 3, **kwargs):
        """
        Initialize H2 environment variant.
        
        Args:
            fixed_explain_duration: Fixed number of turns for Explain option
        """
        super().__init__(*args, **kwargs)
        self.fixed_explain_duration = fixed_explain_duration
        self.explain_turn_count = 0
        
        print(f"[H2] Fixed-duration Explain: {fixed_explain_duration} turns")
        print("     (vs learned terminations in baseline)")
    
    def step(self, action_dict):
        """Override step to enforce fixed-duration Explain."""
        # Extract option
        option_idx = action_dict.get("option", 0)
        available_options = self._get_available_options()
        if option_idx < len(available_options):
            option = available_options[option_idx]
        else:
            option = available_options[0] if available_options else "Explain"
        
        # Force termination if Explain has reached fixed duration
        if option == "Explain":
            self.explain_turn_count += 1
            if self.explain_turn_count >= self.fixed_explain_duration:
                # Force termination
                action_dict["terminate_option"] = True
                self.explain_turn_count = 0
        else:
            # Reset counter when switching away from Explain
            self.explain_turn_count = 0
        
        # Call parent step
        return super().step(action_dict)
    
    def reset(self, seed=None, options=None):
        """Reset environment and Explain counter."""
        obs, info = super().reset(seed=seed, options=options)
        self.explain_turn_count = 0
        return obs, info

