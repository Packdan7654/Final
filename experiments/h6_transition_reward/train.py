"""
Training script for H6 experiment: Transition Reward Shaping

Trains agent with transition reward components disabled to test
if probabilistic transition acceptance alone is sufficient.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import os
import json
from datetime import datetime
from src.training.training_loop import HRLTrainingLoop
from experiments.h6_transition_reward.env import H6TransitionRewardEnv


class H6TrainingLoop(HRLTrainingLoop):
    """Training loop for H6 experiment using transition reward variant."""
    
    def __init__(self, *args, **kwargs):
        """Initialize H6 training loop."""
        # Temporarily store knowledge graph path
        kg_path = kwargs.get('knowledge_graph_path', None)
        
        # Initialize parent without environment
        kwargs['use_actor_critic'] = False  # Prevent env initialization
        super().__init__(*args, **kwargs)
        
        # Replace environment with H6 variant
        max_turns = kwargs.get('max_turns_per_episode', self.max_turns_per_episode)
        self.env = H6TransitionRewardEnv(
            knowledge_graph_path=kg_path,
            max_turns=max_turns
        )
        
        # Reinitialize agent (state dimension unchanged)
        if self.use_actor_critic:
            state_dim = self.env.observation_space.shape[0]
            from src.agent.actor_critic_agent import ActorCriticAgent
            self.agent = ActorCriticAgent(
                state_dim=state_dim,
                options=self.env.options,
                subactions=self.env.subactions,
                hidden_dim=256,
                lstm_hidden_dim=128,
                use_lstm=True,
                device=self.device
            )
            
            from src.training.actor_critic_trainer import ActorCriticTrainer
            learning_rate = kwargs.get('learning_rate', 1e-4)
            gamma = kwargs.get('gamma', 0.99)
            self.trainer = ActorCriticTrainer(
                agent=self.agent,
                learning_rate=learning_rate,
                gamma=gamma,
                device=self.device
            )
        
        self.training_title = "H6 EXPERIMENT: TRANSITION REWARD SHAPING"


def main():
    parser = argparse.ArgumentParser(description='Train H6: Transition Reward')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--name', type=str, default='h6_transition_reward')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Create experiment directory
    from train import create_experiment_folder
    exp_dir, exp_num = create_experiment_folder(args.name, 'major')
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    
    # Save metadata
    metadata = {
        "experiment_number": exp_num,
        "experiment_name": args.name,
        "hypothesis": "H6",
        "variant": "transition_reward_disabled",
        "timestamp": datetime.now().isoformat(),
        "episodes": args.episodes,
        "max_turns_per_episode": args.turns,
        "device": args.device,
        "learning_rate": args.lr,
        "gamma": args.gamma,
    }
    
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 80)
    print("H6 EXPERIMENT: TRANSITION REWARD SHAPING")
    print("=" * 80)
    print(f"Testing: With vs without transition reward components")
    print(f"Episodes: {args.episodes}")
    print("=" * 80)
    print()
    
    # Initialize and run training
    training_loop = H6TrainingLoop(
        max_episodes=args.episodes,
        max_turns_per_episode=args.turns,
        knowledge_graph_path="museum_knowledge_graph.json",
        learning_rate=args.lr,
        gamma=args.gamma,
        use_actor_critic=True,
        device=args.device,
        verbose=args.verbose
    )
    
    training_loop.run_training()
    
    print("\n✅ H6 training complete!")
    print(f"Results: {exp_dir}")


if __name__ == '__main__':
    main()

