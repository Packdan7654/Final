"""
Training script for H1 experiment: Option Structure vs Flat Policy

Trains flat RL variant (no options, primitive actions only) for comparison
with hierarchical baseline.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import json
from datetime import datetime
from src.flat_rl.training_loop import FlatTrainingLoop


def main():
    parser = argparse.ArgumentParser(description='Train H1: Flat Policy (vs Options)')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--name', type=str, default='h1_flat_policy')
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
        "hypothesis": "H1",
        "variant": "flat_policy",
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
    print("H1 EXPERIMENT: FLAT POLICY (vs OPTIONS)")
    print("=" * 80)
    print(f"Testing: Flat policy vs hierarchical options")
    print(f"Episodes: {args.episodes}")
    print("=" * 80)
    print()
    
    # Initialize and run training (uses FlatTrainingLoop)
    training_loop = FlatTrainingLoop(
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
    
    print("\n✅ H1 training complete!")
    print(f"Results: {exp_dir}")


if __name__ == '__main__':
    main()

