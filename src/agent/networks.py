"""
Actor-Critic Neural Networks for Option-Critic Architecture

This implements the Actor-Critic networks as described in Bacon et al. (2017)
"The Option-Critic Architecture" for hierarchical reinforcement learning.

Components:
- Actor: Learns policies π_Ω(ω|s), π_o(a|s), β_o(s)
- Critic: Learns value functions Q_Ω(s,ω), Q_U(s,ω,a), V(s)

The network processes state with DialogueBERT embeddings (149-d):
- Focus vector (9-d)
- History vector (12-d)  
- Intent embedding (64-d from DialogueBERT)
- Context embedding (64-d from DialogueBERT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for Option-Critic Architecture.
    
    Actor learns:
    - π_Ω(ω|s): Option policy
    - π_o(a|s): Intra-option policies  
    - β_o(s): Termination functions
    
    Critic learns:
    - Q_Ω(s,ω): Option-value function
    - Q_U(s,ω,a): Action-value function
    - V(s): State value function
    """
    
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        num_subactions: int,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_options = num_options
        self.num_subactions = num_subactions
        self.use_lstm = use_lstm
        
        # ===== SHARED ENCODER =====
        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=state_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True
            )
            encoder_dim = lstm_hidden_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            encoder_dim = hidden_dim
        
        # ===== ACTOR: POLICIES =====
        
        # Option policy π_Ω(ω|s)
        self.option_policy = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Intra-option policies π_o(a|s) - one per option
        self.intra_option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subactions)
            )
            for _ in range(num_options)
        ])
        
        # Termination functions β_o(s) - one per option
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_options)
        ])
        
        # ===== CRITIC: VALUE FUNCTIONS =====
        
        # Option-value Q_Ω(s,ω)
        self.option_value = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Action-value Q_U(s,ω,a) - one per option
        self.action_value = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_subactions)
            )
            for _ in range(num_options)
        ])
        
        # State value V(s)
        self.state_value = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hidden_state = None
        
    def forward(self, state: torch.Tensor, reset_hidden: bool = False):
        """
        Forward pass through Actor-Critic network.
        
        Args:
            state: (batch, state_dim) or (batch, seq_len, state_dim)
            reset_hidden: Reset LSTM hidden state
            
        Returns:
            Dictionary with actor and critic outputs
        """
        if reset_hidden or self.hidden_state is None:
            self.hidden_state = None
        
        # Encode state
        if self.use_lstm:
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            
            if self.hidden_state is not None:
                encoded, self.hidden_state = self.encoder(state, self.hidden_state)
            else:
                encoded, self.hidden_state = self.encoder(state)
            
            encoded = encoded[:, -1, :]
        else:
            encoded = self.encoder(state)
        
        # ===== ACTOR OUTPUTS =====
        
        # Option policy logits
        option_logits = self.option_policy(encoded)
        
        # Intra-option policy logits
        intra_option_logits = [policy(encoded) for policy in self.intra_option_policies]
        
        # Termination probabilities
        termination_probs = torch.cat([term(encoded) for term in self.termination_functions], dim=-1)
        
        # ===== CRITIC OUTPUTS =====
        
        # Option-values Q_Ω(s,ω)
        option_values = self.option_value(encoded)
        
        # Action-values Q_U(s,ω,a) per option
        action_values = [value(encoded) for value in self.action_value]
        
        # State value V(s)
        state_value = self.state_value(encoded).squeeze(-1)
        
        return {
            'encoded': encoded,
            'option_logits': option_logits,
            'intra_option_logits': intra_option_logits,
            'termination_probs': termination_probs,
            'option_values': option_values,
            'action_values': action_values,
            'state_value': state_value
        }
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state for new episode."""
        self.hidden_state = None

