"""
Train HRL Museum Dialogue Agent

Unified training script with all features:
- Configurable reward weights (per paper formalization)
- Comprehensive evaluation (always enabled)
- Map visualization at specified turn numbers
- Detailed logging and metrics
- Parameterization analysis

Usage:
    python train.py --episodes 600 --device cuda
    python train.py --episodes 200 --w-engagement 1.0 --w-novelty 0.5
    python train.py --episodes 500 --map-turns 10,20,30 --save-map-frames
"""

import argparse
import torch
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from src.training.training_loop import HRLTrainingLoop


def create_experiment_folder(name=None):
    """Create organized experiment folder structure with date-based organization."""
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Create date-based folder structure: training_logs/experiments/YYYYMMDD/
    exp_base = Path("training_logs/experiments")
    date_folder = exp_base / date_str
    date_folder.mkdir(parents=True, exist_ok=True)
    
    # Find next experiment number by looking only in the date folder
    # Each date gets its own experiment numbering starting from 001
    existing = list(date_folder.glob("exp_*"))
    
    if existing:
        numbers = []
        for e in existing:
            parts = e.name.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                numbers.append(int(parts[1]))
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    
    # Create experiment folder inside date folder
    if name:
        exp_name = f"exp_{next_num:03d}_{name}_{timestamp}"
    else:
        exp_name = f"exp_{next_num:03d}_{timestamp}"
    
    exp_dir = date_folder / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "maps").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "detailed_logs").mkdir(exist_ok=True)
    (exp_dir / "parameterization_results").mkdir(exist_ok=True)
    
    return exp_dir, next_num




def main():
    parser = argparse.ArgumentParser(
        description='Train HRL Museum Dialogue Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reward Configuration (per paper.tex Section 4.7, line 634-637):
  R_t = r^eng + r^nov + r^resp + r^trans + r^conclude

Where:
  r^eng_t = dwell_t (engagement from gaze)
  r^nov_t = 0.15 × |new facts at t| (knowledge novelty, scale α = 0.15)
  
Default parameters match paper baseline:
  --w-engagement 1.0              (engagement: r^eng_t = dwell_t × 1.0)
  --novelty-per-fact 0.15         (novelty scale α: r^nov_t = 0.15 × |new facts|)
  --w-responsiveness 0.25          (responsiveness reward scale)
  --w-conclude 0.2                 (conclude bonus per exhibit)
  --w-transition-insufficiency -0.20  (transition penalty when < 2 facts)

Note: Per paper.tex line 637, no separate weights are applied - these are the reward scales themselves.
      Question spam and transition spam are handled at simulator level (reduce dwell time).
      Transition insufficiency is an explicit reward component (3-turn exemption if successful).

Example:
  python train.py --episodes 600 --novelty-per-fact 0.2
        """
    )
    
    # ===== CORE TRAINING ARGUMENTS =====
    parser.add_argument('--episodes', type=int, default=600,
                       help='Number of training episodes (default: 600)')
    parser.add_argument('--turns', type=int, default=40,
                       help='Max turns per episode (default: 40)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training (default: cpu)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (optional identifier)')
    
    # ===== REWARD PARAMETERS (per paper formalization, Section 4.7) =====
    # Note: Per paper.tex line 637, r^eng_t = dwell_t and r^nov_t = 0.15 × |new facts|
    # No weights are applied - these are the reward scales themselves
    
    parser.add_argument('--w-engagement', type=float, default=1.0,
                       help='Engagement reward scale (default: 1.0, per paper r^eng_t = dwell_t)')
    parser.add_argument('--novelty-per-fact', type=float, default=0.15,
                       help='Novelty reward scale α: r^nov_t = α × |new facts| (default: 0.15, per paper.tex line 637)')
    parser.add_argument('--w-responsiveness', type=float, default=0.25,
                       help='Responsiveness reward scale (default: 0.25)')
    parser.add_argument('--w-conclude', type=float, default=0.2,
                       help='Conclude bonus scale per exhibit (default: 0.2)')
    
    # Transition insufficiency penalty parameter (per paper.tex Section 4.7)
    # Note: Transition spam is handled at simulator level (reduces dwell time)
    parser.add_argument('--w-transition-insufficiency', type=float, default=-0.20,
                       help='Transition insufficiency penalty when < 2 facts, scales with fewer facts (default: -0.20)')
    
    # ===== EVALUATION & VISUALIZATION =====
    parser.add_argument('--map-interval', type=int, default=50,
                       help='Save map visualization every N episodes (default: 50, set to 0 to disable)')
    parser.add_argument('--save-map-frames', action='store_true',
                       help='Save map snapshots at EVERY turn (for all episodes)')
    parser.add_argument('--live-map-display', action='store_true',
                       help='Show live map windows during training (default: save only)')
    
    # ===== DEBUGGING & TESTING =====
    parser.add_argument('--show-prompts', action='store_true',
                       help='Show LLM prompts during training (for debugging)')
    parser.add_argument('--force-option', type=str, default=None,
                       choices=['Explain', 'AskQuestion', 'OfferTransition', 'Conclude'],
                       help='Force agent to always choose this option (testing only)')
    parser.add_argument('--force-subaction', type=str, default=None,
                       help='Force agent to always choose this subaction (testing only)')
    parser.add_argument('--enable-live-monitor', action='store_true',
                       help='Enable live training monitor with turn-by-turn visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output mode')
    
    args = parser.parse_args()
    
    # Check for Groq API key - try to load from key.txt if not set
    if "GROQ_API_KEY" not in os.environ:
        key_file = Path("key.txt")
        if key_file.exists():
            try:
                api_key = key_file.read_text().strip()
                os.environ["GROQ_API_KEY"] = api_key
                print("[OK] Loaded Groq API key from key.txt")
            except Exception as e:
                print("=" * 80)
                print("⚠️  WARNING: GROQ_API_KEY not set and failed to read key.txt!")
                print("=" * 80)
                print(f"Error: {e}")
                print("Set it with:")
                print("  Windows PowerShell: $env:GROQ_API_KEY='your_key_here'")
                print("  Linux/Mac: export GROQ_API_KEY='your_key_here'")
                print("  Or create key.txt in project root with your API key")
                print("=" * 80)
                print()
        else:
            print("=" * 80)
            print("⚠️  WARNING: GROQ_API_KEY not set and key.txt not found!")
            print("=" * 80)
            print("Set it with:")
            print("  Windows PowerShell: $env:GROQ_API_KEY='your_key_here'")
            print("  Linux/Mac: export GROQ_API_KEY='your_key_here'")
            print("  Or create key.txt in project root with your API key")
            print("=" * 80)
            print()
    
    # Create experiment folder
    exp_dir, exp_num = create_experiment_folder(args.name)
    
    # Set experiment directory environment variable
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    
    # Save experiment metadata with reward weights
    metadata = {
        "experiment_number": exp_num,
        "experiment_name": args.name or "unnamed",
        "timestamp": datetime.now().isoformat(),
        "episodes": args.episodes,
        "max_turns_per_episode": args.turns,
        "device": args.device,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "reward_parameters": {
            "w_engagement": args.w_engagement,
            "novelty_per_fact": args.novelty_per_fact,
            "w_responsiveness": args.w_responsiveness,
            "w_conclude": args.w_conclude,
            "w_transition_insufficiency": args.w_transition_insufficiency
        },
        "note": "Per paper.tex line 637: r^eng_t = dwell_t, r^nov_t = 0.15 × |new facts| (no separate weights). Question spam and transition spam handled at simulator level (reduce dwell).",
        "map_interval": args.map_interval,
        "save_map_frames": args.save_map_frames
    }
    
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print configuration
    print("=" * 80)
    print("TRAINING HRL MUSEUM DIALOGUE AGENT")
    print("=" * 80)
    print(f"EXPERIMENT {exp_num:03d}: {exp_dir.name}")
    print(f"Architecture: Actor-Critic (per paper.tex)")
    print(f"Options: Explain, Ask, Transition, Conclude")
    print(f"State: 149-dim (Focus + History + DialogueBERT Intent + Context)")
    print(f"LLM: Groq API (Llama 3.1 8B)")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Turns/episode: {args.turns}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Device: {args.device}")
    print()
    print("REWARD PARAMETERS (R_t = r^eng + r^nov + r^resp + r^trans + r^conclude, per paper.tex line 634):")
    print(f"  Engagement:         r^eng_t = dwell_t × {args.w_engagement:.2f}")
    print(f"  Novelty:            r^nov_t = {args.novelty_per_fact:.2f} × |new facts| (per paper.tex line 637)")
    print(f"  Responsiveness:     ±{args.w_responsiveness:.2f} (answer questions / deflect penalty)")
    print(f"  Conclude bonus:     {args.w_conclude:.2f} per exhibit covered")
    print(f"  Transition penalty: {args.w_transition_insufficiency:.2f} (scales with fewer facts, per paper.tex Section 4.7)")
    print()
    print("  Note: Per paper, r^eng_t = dwell_t and r^nov_t = 0.15 × |new facts| (no separate weights)")
    print("        Question spam and transition spam handled at simulator level (reduce dwell).")
    print("        Transition insufficiency is explicit reward component (3-turn exemption if successful).")
    print()
    print("EVALUATION & VISUALIZATION:")
    if args.map_interval > 0:
        print(f"  Map interval:        every {args.map_interval} episodes")
    if args.save_map_frames:
        print(f"  Map frames:         saving EVERY turn (all episodes)")
    print(f"  Evaluation:         ALWAYS ENABLED")
    print("=" * 80)
    print()
    
    # Pass reward parameters to environment via environment variable
    os.environ["HRL_W_ENGAGEMENT"] = str(args.w_engagement)
    os.environ["HRL_NOVELTY_PER_FACT"] = str(args.novelty_per_fact)
    os.environ["HRL_W_RESPONSIVENESS"] = str(args.w_responsiveness)
    os.environ["HRL_W_CONCLUDE"] = str(args.w_conclude)
    os.environ["HRL_W_TRANSITION_INSUFFICIENCY"] = str(args.w_transition_insufficiency)
    
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
        force_subaction=args.force_subaction,
        enable_live_monitor=args.enable_live_monitor,
        save_metrics=True,  # Always enabled
        enable_map_viz=args.map_interval > 0 or args.save_map_frames,
        save_map_frames=args.save_map_frames,
        live_map_display=args.live_map_display,
        map_interval=args.map_interval,
        verbose=args.verbose
    )
    
    # Train
    print("Starting training...")
    print()
    training_loop.run_training()
    
    # Save model
    model_path = exp_dir / "models" / "trained_agent.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    
    checkpoint = {
        'agent_state_dict': training_loop.agent.network.state_dict(),
        'options': training_loop.env.options,
        'subactions': training_loop.env.subactions,
        'state_dim': training_loop.env.observation_space.shape[0],
        'config': {
            'episodes': args.episodes,
            'turns': args.turns,
            'lr': args.lr,
            'gamma': args.gamma,
            'reward_parameters': metadata['reward_parameters']
        },
        'timestamp': datetime.now().isoformat(),
        'total_episodes': training_loop.total_episodes,
        'avg_reward': sum(training_loop.episode_rewards) / len(training_loop.episode_rewards) 
                     if training_loop.episode_rewards else 0
    }
    
    torch.save(checkpoint, model_path)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Episodes: {training_loop.total_episodes}")
    print(f"✓ Avg reward: {checkpoint['avg_reward']:.3f}")
    print("=" * 80)
    
    # ALWAYS run evaluation (mandatory)
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION (mandatory)")
    print("=" * 80)
    
    try:
        from create_evaluation_plots import HRLEvaluationPlotter
        
        plotter = HRLEvaluationPlotter(exp_dir)
        plotter.load_data()
        plotter.generate_all_plots()
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"✓ All plots saved to: {exp_dir / 'evaluation'}")
        print(f"✓ Summary saved to: {exp_dir / 'evaluation' / 'EVALUATION_SUMMARY.txt'}")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate evaluation plots: {e}")
        print("   This is a critical error - evaluation is mandatory!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run parameterization analysis
    print("\n" + "=" * 80)
    print("RUNNING PARAMETERIZATION ANALYSIS")
    print("=" * 80)
    
    try:
        from src.utils.parameterization_analyzer import ParameterizationAnalyzer
        
        analyzer = ParameterizationAnalyzer(exp_dir)
        analyzer.generate_full_report()
        
        print(f"✓ Analysis saved to: {exp_dir / 'parameterization_results'}")
        print("=" * 80)
    except Exception as e:
        print(f"⚠️  Warning: Parameterization analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final summary
    summary = {
        **metadata,
        "status": "completed",
        "completion_time": datetime.now().isoformat(),
        "total_episodes": training_loop.total_episodes,
        "avg_reward": checkpoint['avg_reward']
    }
    
    with open(exp_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_num:03d} SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {exp_dir}/")
    print(f"  - Models:           {exp_dir / 'models'}")
    print(f"  - Logs:             {exp_dir / 'logs'}")
    print(f"  - Maps:             {exp_dir / 'maps'}")
    print(f"  - Detailed logs:    {exp_dir / 'detailed_logs'}")
    print(f"  - Evaluation:       {exp_dir / 'evaluation'}")
    print(f"  - Parameterization: {exp_dir / 'parameterization_results'}")
    print(f"  - Checkpoints:      {exp_dir / 'checkpoints'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
