"""
Simple Test Script for Trained HRL Museum Agent

Loads the trained model and runs a few test episodes to evaluate quality.
"""

import torch
import numpy as np
from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent

def load_and_test_model():
    """Load trained model and run test"""
    print("=" * 80)
    print("TESTING TRAINED HRL MUSEUM AGENT")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint = torch.load("models/trained_agent.pt", map_location='cpu')
    
    print(f"\n[Model Info]")
    print(f"  Trained: {checkpoint.get('timestamp', 'N/A')}")
    print(f"  Episodes: {checkpoint.get('total_episodes', 'N/A')}")
    print(f"  Avg Reward: {checkpoint.get('avg_reward', 0):.3f}")
    print(f"  State Dim: {checkpoint.get('state_dim', 'N/A')}")
    
    # Initialize environment
    env = MuseumDialogueEnv(
        knowledge_graph_path="museum_knowledge_graph.json"
    )
    
    # Initialize agent
    agent = ActorCriticAgent(
        state_dim=checkpoint['state_dim'],
        options=checkpoint['options'],
        subactions=checkpoint['subactions']
    )
    
    # Load weights
    agent.network.load_state_dict(checkpoint['agent_state_dict'])
    agent.network.eval()
    
    print(f"\n[Agent Stats]")
    print(f"  Parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"  Options: {checkpoint['options']}")
    print(f"  Subactions: {list(checkpoint['subactions'].keys())}")
    
    # Run a simple test episode
    print("\n" + "=" * 80)
    print("RUNNING TEST EPISODE (1 episode, 10 turns)")
    print("=" * 80 + "\n")
    
    state, _ = env.reset()
    total_reward = 0
    
    for turn in range(10):
        # Get available options and subactions
        available_options = env._get_available_options()
        available_subactions_dict = {opt: env._get_available_subactions(opt) for opt in available_options}
        
        # Select action using trained policy
        with torch.no_grad():
            action_dict = agent.select_action(
                state, available_options, available_subactions_dict
            )
        
        # Get option/subaction names
        option_idx = action_dict['option']
        subaction_idx = action_dict['subaction']
        terminated = action_dict['terminated']
        
        option_name = available_options[option_idx]
        available_subs = available_subactions_dict[option_name]
        subaction_name = available_subs[subaction_idx] if subaction_idx < len(available_subs) else "Unknown"
        
        print(f"Turn {turn+1}:")
        print(f"  Option: {option_name} | Subaction: {subaction_name}")
        
        # Take step (this might fail since we need proper integration)
        try:
            # Build action dict for environment step
            env_action = {
                'option': option_idx,
                'subaction': subaction_idx,
                'terminate_option': terminated
            }
            next_state, reward, done, truncated, info = env.step(env_action)
            
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            
            total_reward += reward
            state = next_state
            
            if done:
                print(f"\nEpisode finished early at turn {turn+1}")
                break
        except Exception as e:
            print(f"  Error during step: {e}")
            break
    
    print(f"\n" + "=" * 80)
    print(f"TEST COMPLETE")
    print(f"=" * 80)
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Avg Reward/Turn: {total_reward/(turn+1):.3f}")
    
    # Print policy statistics
    print(f"\n[Policy Network Stats]")
    print(f"  Network architecture: Actor-Critic with Options")
    print(f"  Hidden layers: {agent.network.hidden_dim} units")
    if hasattr(agent.network, 'use_lstm') and agent.network.use_lstm:
        print(f"  LSTM: Enabled ({agent.network.lstm_hidden_dim} units)")
    print(f"  Output: {len(checkpoint['options'])} options x subactions + termination")
    
    return checkpoint, total_reward


if __name__ == "__main__":
    checkpoint, reward = load_and_test_model()
    print("\nTest completed successfully!")

