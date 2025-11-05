"""
Training Monitor for HRL Museum Dialogue Agent

This module provides simple logging and statistics tracking for the training process.
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque

class TrainingMonitor:
    """Simple training monitor for logging and statistics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.training_history = deque(maxlen=max_history)
        self.current_episode = 0
        self.current_turn = 0
        self.episode_rewards = []
        self.option_usage = {}
        self.subaction_usage = {}
        
    def update_training_step(self, 
                           state: np.ndarray,
                           action: Dict[str, Any],
                           reward: float,
                           done: bool,
                           info: Dict[str, Any],
                           simulator_data: Optional[Dict[str, Any]] = None):
        """Update monitor with new training step data"""
        timestamp = datetime.now()
        
        # Store training data
        step_data = {
            'timestamp': timestamp,
            'episode': self.current_episode,
            'turn': self.current_turn,
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'done': done,
            'info': info.copy(),
            'simulator_data': simulator_data
        }
        
        self.training_history.append(step_data)
        
        # Update usage statistics - convert to names if indices
        option = info.get('option', 'Unknown')
        subaction = info.get('subaction', 'Unknown')
        
        # If option is a string name, use it; otherwise keep as Unknown for now
        # (will be converted during analysis if indices are present)
        self.option_usage[option] = self.option_usage.get(option, 0) + 1
        self.subaction_usage[subaction] = self.subaction_usage.get(subaction, 0) + 1
        
        # Update turn counter
        self.current_turn += 1
        
        # Check if episode ended
        if done:
            self.current_episode += 1
            self.current_turn = 0
            
            # Calculate episode reward
            episode_reward = sum([step['reward'] for step in self.training_history 
                                if step['episode'] == self.current_episode - 1])
            self.episode_rewards.append(episode_reward)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training statistics"""
        if not self.training_history:
            return {
                'total_episodes': 0,
                'total_turns': 0,
                'average_reward_per_turn': 0.0,
                'average_reward_per_episode': 0.0,
                'best_episode_reward': 0.0,
                'most_used_option': 'None',
                'most_used_subaction': 'None',
                'option_usage': {},
                'subaction_usage': {}
            }
        
        total_turns = len(self.training_history)
        total_reward = sum(step['reward'] for step in self.training_history)
        
        # Find most used option and subaction
        most_used_option = max(self.option_usage.items(), key=lambda x: x[1])[0] if self.option_usage else 'None'
        most_used_subaction = max(self.subaction_usage.items(), key=lambda x: x[1])[0] if self.subaction_usage else 'None'
        
        return {
            'total_episodes': self.current_episode,
            'total_turns': total_turns,
            'average_reward_per_turn': total_reward / total_turns if total_turns > 0 else 0.0,
            'average_reward_per_episode': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'best_episode_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'most_used_option': most_used_option,
            'most_used_subaction': most_used_subaction,
            'option_usage': self.option_usage.copy(),
            'subaction_usage': self.subaction_usage.copy()
        }
    
    def print_current_status(self):
        """Print current training status"""
        if not self.training_history:
            print("No training data available")
            return
        
        latest_step = self.training_history[-1]
        info = latest_step['info']
        
        print(f"\n📊 Training Status (Episode {self.current_episode}, Turn {self.current_turn}):")
        print(f"   Option: {info.get('option', 'Unknown')}")
        print(f"   Subaction: {info.get('subaction', 'Unknown')}")
        print(f"   Reward: {latest_step['reward']:.4f}")
        print(f"   Agent: {info.get('agent_utterance', 'None')[:50]}...")
        print(f"   Focus: {info.get('current_focus', 0)}")
        print(f"   Facts Shared: {info.get('facts_shared', 0)}")
        print(f"   Exhibits Covered: {info.get('exhibits_covered', 0)}")
        
        if latest_step['simulator_data']:
            sim_data = latest_step['simulator_data']
            print(f"   User: {sim_data.get('utterance', 'None')}")
            print(f"   AOI: {sim_data.get('aoi', 'None')}")
    
    def close(self):
        """Clean up monitor resources"""
        pass  # No resources to clean up in simplified version
