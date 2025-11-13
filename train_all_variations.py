"""
Master Training Script for All Model Variations

Trains all model variations (baseline, h1, h3, h5, h6, h7) sequentially with
identical parameters except for the specific variation change.

Usage:
    # Train all variations
    python train_all_variations.py --episodes 600 --map-interval 10
    
    # Train specific variations
    python train_all_variations.py --variations baseline h1 h3 --episodes 600
    
    # Test run (5 episodes, map interval 2)
    python train_all_variations.py --episodes 5 --map-interval 2 --test-mode
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Baseline parameters (used for all variations)
BASELINE_PARAMS = {
    'episodes': 500,
    'turns': 40,
    'lr': 1e-4,
    'gamma': 0.99,
    'device': 'cuda',
    'w_engagement': 1.0,
    'novelty_per_fact': 0.25,
    'w_responsiveness': 0.25,
    'w_conclude': 0.2,
    'w_transition_insufficiency': -0.20,
    'map_interval': 10,
    'save_map_frames': False,
}


# Variation configurations
VARIATIONS = {
    'baseline': {
        'name': 'baseline',
        'script': 'train.py',
        'args': ['--mode', 'hrl'],
        'description': 'Baseline: Hierarchical Option-Critic with Standard BERT',
        'requires_training': True,
    },
    'h1': {
        'name': 'h1_flat_policy',
        'script': 'experiments/h1_option_structure/train.py',
        'args': [],
        'description': 'H1: Flat Actor-Critic (no hierarchical structure)',
        'requires_training': True,
    },
    'h3': {
        'name': 'h3_minimal_prompts',
        'script': 'experiments/h3_prompt_headers/train.py',
        'args': [],
        'description': 'H3: Minimal Prompts (no structured headers)',
        'requires_training': True,
    },
    'h5': {
        'name': 'h5_state_ablation',
        'script': 'experiments/h5_state_ablation/train.py',
        'args': [],
        'description': 'H5: State Ablation (dialogue-act-only state)',
        'requires_training': True,
    },
    'h6': {
        'name': 'h6_transition_reward',
        'script': 'experiments/h6_transition_reward/train.py',
        'args': [],
        'description': 'H6: No Transition Rewards',
        'requires_training': True,
    },
    'h7': {
        'name': 'h7_hybrid_bert',
        'script': 'experiments/h7_hybrid_bert/train.py',
        'args': [],
        'description': 'H7: Hybrid BERT (standard for intent, DialogueBERT for context)',
        'requires_training': True,
    },
}


def print_header(text: str, char: str = '='):
    """Print a formatted header."""
    print("\n" + char * 80)
    print(text)
    print(char * 80 + "\n")


def print_variation_info(variation: str, config: Dict):
    """Print information about a variation."""
    print(f"  [{variation.upper()}] {config['description']}")
    print(f"      Script: {config['script']}")
    print(f"      Name: {config['name']}")


def build_command(variation: str, config: Dict, params: Dict, test_mode: bool = False) -> List[str]:
    """
    Build command to run a variation.
    
    Args:
        variation: Variation key (baseline, h1, etc.)
        config: Variation configuration
        params: Training parameters
        test_mode: Whether in test mode (reduces output)
        
    Returns:
        Command as list of strings
    """
    script_path = Path(config['script'])
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    cmd = [sys.executable, str(script_path)]
    
    # Add common arguments
    cmd.extend([
        '--episodes', str(params['episodes']),
        '--turns', str(params['turns']),
        '--lr', str(params['lr']),
        '--gamma', str(params['gamma']),
        '--device', params['device'],
        '--name', config['name'],
    ])
    
    # Add experiment-type major for all variations (saves to major_results)
    if variation == 'baseline':
        # train.py supports --experiment-type
        cmd.extend(['--experiment-type', 'major'])
    
    # Add variation-specific arguments
    cmd.extend(config['args'])
    
    # Add reward parameters (for baseline only, others inherit from env)
    if variation == 'baseline':
        cmd.extend([
            '--w-engagement', str(params['w_engagement']),
            '--novelty-per-fact', str(params['novelty_per_fact']),
            '--w-responsiveness', str(params['w_responsiveness']),
            '--w-conclude', str(params['w_conclude']),
            '--w-transition-insufficiency', str(params['w_transition_insufficiency']),
        ])
        
        # Add map interval (only baseline train.py supports this)
        if params.get('map_interval', 0) > 0:
            cmd.extend(['--map-interval', str(params['map_interval'])])
        
        # Add map frames flag (only baseline train.py supports this)
        if params.get('save_map_frames', False):
            cmd.append('--save-map-frames')
    
    # Don't add verbose flag - let users control it explicitly
    # Non-verbose mode shows clean episode progress only
    
    return cmd


def run_variation(variation: str, config: Dict, params: Dict, test_mode: bool = False) -> bool:
    """
    Run a single variation.
    
    Args:
        variation: Variation key
        config: Variation configuration
        params: Training parameters
        test_mode: Whether in test mode
        
    Returns:
        True if successful, False otherwise
    """
    print_header(f"TRAINING: {variation.upper()} - {config['description']}")
    
    if not config['requires_training']:
        print(f"  [SKIP] Skipping {variation}: Does not require training")
        return True
    
    # Build command
    try:
        cmd = build_command(variation, config, params, test_mode)
    except FileNotFoundError as e:
        print(f"  [ERROR] Error: {e}")
        return False
    
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working directory: {project_root}")
    print()
    
    # Run training
    start_time = time.time()
    try:
        # Set environment variable to trigger major_results organization
        env = os.environ.copy()
        env['ORGANIZE_TO_MAJOR_RESULTS'] = 'true'
        
        # Always show output in real-time so we can see what's happening
        # Use Popen for real-time streaming output
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,  # Line buffered
            env=env  # Pass environment with flag
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n  [OK] {variation.upper()} completed successfully!")
        print(f"  Time: {hours}h {minutes}m {seconds}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  [FAILED] {variation.upper()} failed!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  stdout (last 1000 chars):\n{e.stdout[-1000:]}")
        if e.stderr:
            print(f"  stderr (last 1000 chars):\n{e.stderr[-1000:]}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] {variation.upper()} encountered an unexpected error!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {variation.upper()} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train all model variations sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all variations with default parameters
  python train_all_variations.py
  
  # Train specific variations
  python train_all_variations.py --variations baseline h1 h3
  
  # Test run (5 episodes, map interval 2)
  python train_all_variations.py --episodes 5 --map-interval 2 --test-mode
  
  # Custom parameters
  python train_all_variations.py --episodes 600 --map-interval 10 --device cuda
        """
    )
    
    # Variation selection
    parser.add_argument(
        '--variations',
        nargs='+',
        choices=list(VARIATIONS.keys()) + ['all'],
        default=['all'],
        help='Variations to train (default: all)'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=BASELINE_PARAMS['episodes'],
        help=f"Number of episodes (default: {BASELINE_PARAMS['episodes']})"
    )
    parser.add_argument(
        '--turns',
        type=int,
        default=BASELINE_PARAMS['turns'],
        help=f"Max turns per episode (default: {BASELINE_PARAMS['turns']})"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=BASELINE_PARAMS['lr'],
        help=f"Learning rate (default: {BASELINE_PARAMS['lr']})"
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=BASELINE_PARAMS['gamma'],
        help=f"Discount factor (default: {BASELINE_PARAMS['gamma']})"
    )
    parser.add_argument(
        '--device',
        type=str,
        default=BASELINE_PARAMS['device'],
        choices=['cpu', 'cuda'],
        help=f"Device (default: {BASELINE_PARAMS['device']})"
    )
    
    # Map visualization
    parser.add_argument(
        '--map-interval',
        type=int,
        default=BASELINE_PARAMS['map_interval'],
        help=f"Map visualization interval (default: {BASELINE_PARAMS['map_interval']}, 0 to disable)"
    )
    parser.add_argument(
        '--save-map-frames',
        action='store_true',
        help="Save map frames for every turn"
    )
    
    # Test mode
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help="Test mode: reduces output, faster execution"
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help="Skip confirmation prompt (useful for overnight runs)"
    )
    
    args = parser.parse_args()
    
    # Determine which variations to run
    if 'all' in args.variations:
        variations_to_run = [v for v in VARIATIONS.keys() if VARIATIONS[v]['requires_training']]
    else:
        variations_to_run = [v for v in args.variations if VARIATIONS[v]['requires_training']]
    
    # Build parameters dict
    params = {
        'episodes': args.episodes,
        'turns': args.turns,
        'lr': args.lr,
        'gamma': args.gamma,
        'device': args.device,
        'w_engagement': BASELINE_PARAMS['w_engagement'],
        'novelty_per_fact': BASELINE_PARAMS['novelty_per_fact'],
        'w_responsiveness': BASELINE_PARAMS['w_responsiveness'],
        'w_conclude': BASELINE_PARAMS['w_conclude'],
        'w_transition_insufficiency': BASELINE_PARAMS['w_transition_insufficiency'],
        'map_interval': args.map_interval,
        'save_map_frames': args.save_map_frames,
    }
    
    # Print header
    print_header("MASTER TRAINING SCRIPT - ALL MODEL VARIATIONS")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total variations: {len(variations_to_run)}")
    print(f"Test mode: {args.test_mode}")
    print()
    
    # Print configuration
    print("CONFIGURATION:")
    print(f"  Episodes: {params['episodes']}")
    print(f"  Turns/episode: {params['turns']}")
    print(f"  Learning rate: {params['lr']}")
    print(f"  Gamma: {params['gamma']}")
    print(f"  Device: {params['device']}")
    print(f"  Map interval: {params['map_interval']}")
    print()
    
    # Print variations to run
    print("VARIATIONS TO TRAIN:")
    for var in variations_to_run:
        print_variation_info(var, VARIATIONS[var])
    print()
    
    # Confirm (unless test mode or no-confirm flag)
    if not args.test_mode and not args.no_confirm:
        response = input("Proceed with training? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run each variation
    results = {}
    overall_start = time.time()
    
    for i, variation in enumerate(variations_to_run, 1):
        print_header(f"VARIATION {i}/{len(variations_to_run)}", char='-')
        
        config = VARIATIONS[variation]
        success = run_variation(variation, config, params, test_mode=args.test_mode)
        results[variation] = success
        
        if not success:
            print(f"\n[WARNING] {variation.upper()} failed. Continuing with next variation...")
        
        # Small delay between variations
        if i < len(variations_to_run):
            time.sleep(2)
    
    # Print summary
    print_header("TRAINING SUMMARY")
    
    total_time = time.time() - overall_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("RESULTS:")
    for variation, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {variation.upper()}: {status}")
    
    # Count successes
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)
    
    print()
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    
    if success_count == total_count:
        print("\n[SUCCESS] All variations completed successfully!")
        print("\nResults are organized in major_results/ directory:")
        for variation in variations_to_run:
            model_name = VARIATIONS[variation]['name']
            print(f"  - {model_name}: major_results/{model_name}/")
    else:
        print(f"\n[WARNING] {total_count - success_count} variation(s) failed. Check logs above.")
    
    print()


if __name__ == '__main__':
    main()
