"""
Train HRL Museum Dialogue Agent

Trains the Actor-Critic agent with options (Explain, Ask, Transition, Conclude)
as specified in paper.tex.

LLM Configuration:
- Default: Phi-2 (2.7B) via Ollama - fast and efficient
- To use Mistral (larger, slower):
    export HRL_LLM_MODEL=mistral
- To use TinyLLaMA (smaller, faster):
    export HRL_LLM_MODEL=tinyllama

Usage:
    python train.py --episodes 200
    python train.py --episodes 500 --device cuda
"""

import argparse
import torch
import os
from datetime import datetime
from src.training.training_loop import HRLTrainingLoop


def main():
    parser = argparse.ArgumentParser(description='Train HRL Museum Dialogue Agent')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--turns', type=int, default=30,
                       help='Max turns per episode')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=str, default='models/trained_agent.pt',
                       help='Output path for trained model')
    parser.add_argument('--show-prompts', action='store_true',
                       help='Show LLM prompts during training (for debugging)')
    parser.add_argument('--force-option', type=str, default=None,
                       choices=['Explain', 'AskQuestion', 'OfferTransition', 'Conclude'],
                       help='Force agent to always choose this option (testing only)')
    parser.add_argument('--force-subaction', type=str, default=None,
                       help='Force agent to always choose this subaction (testing only). ' +
                            'Examples: ExplainNewFact, RepeatFact, ClarifyFact, AskOpinion, ' +
                            'AskMemory, AskClarification, TransitionToExhibit, WrapUp')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING HRL MUSEUM DIALOGUE AGENT")
    print("=" * 80)
    print(f"Architecture: Actor-Critic (per paper.tex)")
    print(f"Options: Explain, Ask, Transition, Conclude")
    print(f"State: 149-d (Focus + History + DialogueBERT Intent + Context)")
    print(f"LLM: Mistral (via Ollama)")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Turns/episode: {args.turns}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    if args.force_option or args.force_subaction:
        print("=" * 80)
        print("⚠️  TESTING MODE (Forced Actions):")
        if args.force_option:
            print(f"  Force Option: {args.force_option}")
        if args.force_subaction:
            print(f"  Force Subaction: {args.force_subaction}")
    print("=" * 80)
    print()
    
    # Initialize training loop
    training_loop = HRLTrainingLoop(
        max_episodes=args.episodes,
        max_turns_per_episode=args.turns,
        knowledge_graph_path="museum_knowledge_graph.json",
        learning_rate=args.lr,
        gamma=args.gamma,
        use_actor_critic=True,
        device=args.device,
        turn_delay=0.0,
        show_prompts=args.show_prompts,
        force_option=args.force_option,
        force_subaction=args.force_subaction
    )
    
    # Train
    print("Starting training...")
    print()
    training_loop.run_training()
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    checkpoint = {
        'agent_state_dict': training_loop.agent.network.state_dict(),
        'options': training_loop.env.options,
        'subactions': training_loop.env.subactions,
        'state_dim': training_loop.env.observation_space.shape[0],
        'config': {
            'episodes': args.episodes,
            'turns': args.turns,
            'lr': args.lr,
            'gamma': args.gamma
        },
        'timestamp': datetime.now().isoformat(),
        'total_episodes': training_loop.total_episodes,
        'avg_reward': sum(training_loop.episode_rewards) / len(training_loop.episode_rewards) 
                     if training_loop.episode_rewards else 0
    }
    
    torch.save(checkpoint, args.output)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Model saved: {args.output}")
    print(f"✓ Episodes: {training_loop.total_episodes}")
    print(f"✓ Avg reward: {checkpoint['avg_reward']:.3f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
