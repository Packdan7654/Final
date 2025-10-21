"""
Actor-Critic Training Algorithm for Option-Critic

This implements standard Actor-Critic learning with TD error for training
the Option-Critic architecture. This is simpler and more direct than PPO.

Algorithm:
1. Collect experience (s, ω, a, r, s')
2. Compute TD error: δ = r + γV(s') - V(s)
3. Update Critic to minimize TD error
4. Update Actor using TD error as advantage
5. Update termination functions

References:
- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Bacon et al. (2017): The Option-Critic Architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import os


class ActorCriticTrainer:
    """
    Actor-Critic trainer for Option-Critic agent.
    
    Uses TD(0) learning with:
    - Value function updates (Critic)
    - Policy gradient updates (Actor)
    - Termination function updates
    """
    
    def __init__(
        self,
        agent,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        termination_reg: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize Actor-Critic trainer.
        
        Args:
            agent: ActorCriticAgent instance
            learning_rate: Learning rate
            gamma: Discount factor
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy regularization
            termination_reg: Termination regularization
            max_grad_norm: Gradient clipping threshold
            device: 'cpu' or 'cuda'
        """
        self.agent = agent
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.termination_reg = termination_reg
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Single optimizer for all parameters
        self.optimizer = optim.Adam(
            self.agent.network.parameters(),
            lr=learning_rate
        )
        
        # Training statistics
        self.stats = defaultdict(list)
        
    def update(
        self,
        states: List[np.ndarray],
        options: List[int],
        subactions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ) -> Dict[str, float]:
        """
        Update agent using Actor-Critic algorithm.
        
        Args:
            states: List of states
            options: List of selected options
            subactions: List of selected subactions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            
        Returns:
            Training statistics
        """
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        options_t = torch.LongTensor(options).to(self.device)
        subactions_t = torch.LongTensor(subactions).to(self.device)
        
        # Reset LSTM hidden state before batch forward pass
        self.agent.network.hidden_state = None
        
        # Forward pass for current states
        outputs = self.agent.network.forward(states_t)
        
        # Forward pass for next states (for bootstrapping)
        with torch.no_grad():
            next_outputs = self.agent.network.forward(next_states_t)
            next_values = next_outputs['state_value']
        
        # ===== CRITIC LOSS (TD Error) =====
        current_values = outputs['state_value']
        targets = rewards_t + self.gamma * next_values * (1.0 - dones_t)
        value_loss = F.mse_loss(current_values, targets.detach())
        
        # TD error as advantage
        advantages = (targets - current_values).detach()
        
        # ===== ACTOR LOSS (Policy Gradient) =====
        
        # Option policy loss
        option_logits = outputs['option_logits']
        option_log_probs = F.log_softmax(option_logits, dim=-1)
        selected_option_log_probs = option_log_probs.gather(1, options_t.unsqueeze(1)).squeeze(1)
        option_policy_loss = -(selected_option_log_probs * advantages).mean()
        
        # Option entropy
        option_probs = F.softmax(option_logits, dim=-1)
        option_entropy = -(option_probs * option_log_probs).sum(dim=-1).mean()
        
        # Intra-option policy loss
        subaction_policy_loss = 0.0
        subaction_entropy = 0.0
        
        for i in range(len(states)):
            opt_idx = options[i]
            sub_idx = subactions[i]
            
            # Get subaction logits for this option
            sub_logits = outputs['intra_option_logits'][opt_idx][i:i+1]
            sub_log_probs = F.log_softmax(sub_logits, dim=-1)
            sub_log_prob = sub_log_probs[0, sub_idx]
            
            # Policy gradient
            subaction_policy_loss -= sub_log_prob * advantages[i]
            
            # Entropy
            sub_probs = F.softmax(sub_logits, dim=-1)
            subaction_entropy -= (sub_probs * sub_log_probs).sum()
        
        subaction_policy_loss = subaction_policy_loss / len(states)
        subaction_entropy = subaction_entropy / len(states)
        
        # ===== TERMINATION LOSS =====
        # Encourage termination when advantage is negative
        termination_probs = outputs['termination_probs']
        
        termination_loss = 0.0
        for i in range(len(states)):
            opt_idx = options[i]
            term_prob = termination_probs[i, opt_idx]
            
            # If advantage < 0, encourage termination
            # If advantage > 0, discourage termination
            if advantages[i] < 0:
                termination_loss -= torch.log(term_prob + 1e-10)
            else:
                termination_loss -= torch.log(1.0 - term_prob + 1e-10)
        
        termination_loss = termination_loss / len(states)
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.value_loss_coef * value_loss +
            option_policy_loss +
            subaction_policy_loss -
            self.entropy_coef * (option_entropy + subaction_entropy) +
            self.termination_reg * termination_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Statistics
        stats = {
            'value_loss': value_loss.item(),
            'policy_loss': (option_policy_loss + subaction_policy_loss).item(),
            'entropy': (option_entropy + subaction_entropy).item(),
            'termination_loss': termination_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': current_values.mean().item()
        }
        
        for k, v in stats.items():
            self.stats[k].append(v)
        
        return stats
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        torch.save({
            'episode': episode,
            'network': self.agent.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': dict(self.stats)
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = defaultdict(list, checkpoint['stats'])
        return checkpoint['episode']

