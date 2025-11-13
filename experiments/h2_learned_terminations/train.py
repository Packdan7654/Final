"""
Training script for H2: Fixed-Duration Options

Trains agent with fixed-duration Explain option (no learned terminations).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import json
from datetime import datetime
from src.training.training_loop import HRLTrainingLoop
from experiments.h2_learned_terminations.env import H2FixedDurationEnv


class H2TrainingLoop(HRLTrainingLoop):
    """Training loop for H2 with fixed-duration options."""
    
    def __init__(self, *args, fixed_explain_duration: int = 3, **kwargs):
        kg_path = kwargs.get('knowledge_graph_path', None)
        kwargs['use_actor_critic'] = False
        super().__init__(*args, **kwargs)
        
        max_turns = kwargs.get('max_turns_per_episode', self.max_turns_per_episode)
        self.env = H2FixedDurationEnv(
            knowledge_graph_path=kg_path,
            max_turns=max_turns,
            fixed_explain_duration=fixed_explain_duration
        )
        
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
        
        self.training_title = "H2 EXPERIMENT: FIXED-DURATION OPTIONS"


def main():
    parser = argparse.ArgumentParser(description='Train H2: Fixed-Duration Options')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--name', type=str, default='h2_fixed_duration')
    parser.add_argument('--fixed-duration', type=int, default=3,
                       help='Fixed duration for Explain option')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    from train import create_experiment_folder
    exp_dir, exp_num = create_experiment_folder(args.name, 'major')
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    
    metadata = {
        "experiment_number": exp_num,
        "experiment_name": args.name,
        "hypothesis": "H2",
        "variant": "fixed_duration_options",
        "fixed_explain_duration": args.fixed_duration,
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
    print("H2 EXPERIMENT: FIXED-DURATION OPTIONS")
    print("=" * 80)
    print(f"Testing: Fixed-duration ({args.fixed_duration} turns) vs learned terminations")
    print(f"Episodes: {args.episodes}")
    print("=" * 80)
    print()
    
    training_loop = H2TrainingLoop(
        max_episodes=args.episodes,
        max_turns_per_episode=args.turns,
        knowledge_graph_path="museum_knowledge_graph.json",
        learning_rate=args.lr,
        gamma=args.gamma,
        use_actor_critic=True,
        device=args.device,
        fixed_explain_duration=args.fixed_duration,
        verbose=args.verbose
    )
    
    training_loop.run_training()
    
    print("\n✅ H2 training complete!")
    print(f"Results: {exp_dir}")


if __name__ == '__main__':
    main()

