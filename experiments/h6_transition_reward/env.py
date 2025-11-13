"""
H6 Environment Variant: Transition Reward Toggle

Extends base MuseumDialogueEnv to toggle transition reward components
for hypothesis testing.

Variant: Removes transition insufficiency/sufficiency rewards,
testing if probabilistic transition acceptance alone is sufficient.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.env import MuseumDialogueEnv


class H6TransitionRewardEnv(MuseumDialogueEnv):
    """
    Environment variant for H6 hypothesis: transition reward shaping.
    
    Removes explicit transition insufficiency/sufficiency rewards,
    keeping only probabilistic transition acceptance from simulator.
    Tests if reward shaping improves transition timing.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize H6 environment variant."""
        super().__init__(*args, **kwargs)
        
        # H6: Disable transition reward components
        self.use_transition_rewards = False
        
        print("[H6] Transition reward shaping: DISABLED")
        print("     (Testing probabilistic acceptance alone)")
    
    def step(self, action_dict):
        """
        Override step to disable transition reward components.
        """
        # Call parent step
        obs, reward, done, truncated, info = super().step(action_dict)
        
        # H6: Remove transition reward components from total reward
        if not self.use_transition_rewards:
            # Subtract transition rewards that were added
            transition_penalty = info.get("reward_transition_insufficiency", 0.0)
            transition_sufficiency = info.get("reward_transition_sufficiency", 0.0)
            transition_frequency = info.get("reward_transition_frequency", 0.0)
            
            # Adjust reward
            reward_adjustment = -(transition_penalty + transition_sufficiency + transition_frequency)
            reward += reward_adjustment
            
            # Update info to reflect H6 variant
            info['h6_transition_rewards_removed'] = abs(reward_adjustment)
            info['reward_transition_insufficiency'] = 0.0
            info['reward_transition_sufficiency'] = 0.0
            info['reward_transition_frequency'] = 0.0
        
        return obs, reward, done, truncated, info

