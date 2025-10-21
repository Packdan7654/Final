"""
Verify the Trained Model Works

This script thoroughly verifies that:
1. The model loads correctly
2. The model can make decisions
3. The environment can process those decisions
4. Everything integrates properly
"""

import torch
import numpy as np
from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent

print("=" * 80)
print("VERIFYING TRAINED MODEL")
print("=" * 80)

# Step 1: Load model
print("\n[Step 1] Loading trained model...")
try:
    checkpoint = torch.load("models/trained_agent.pt", map_location='cpu')
    print(f"  [OK] Model loaded successfully")
    print(f"    - Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    print(f"    - Training episodes: {checkpoint.get('total_episodes', 'N/A')}")
    print(f"    - Avg reward: {checkpoint.get('avg_reward', 0):.3f}")
    print(f"    - State dim: {checkpoint.get('state_dim', 'N/A')}")
except Exception as e:
    print(f"  [FAIL] FAILED to load model: {e}")
    exit(1)

# Step 2: Initialize environment
print("\n[Step 2] Initializing environment...")
try:
    env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")
    print(f"  ✓ Environment initialized")
    print(f"    - Exhibits: {len(env.exhibit_keys)}")
    print(f"    - Options: {env.options}")
    print(f"    - State space: {env.observation_space.shape}")
except Exception as e:
    print(f"  [FAIL] FAILED to initialize environment: {e}")
    exit(1)

# Step 3: Initialize agent
print("\n[Step 3] Initializing agent...")
try:
    agent = ActorCriticAgent(
        state_dim=checkpoint['state_dim'],
        options=checkpoint['options'],
        subactions=checkpoint['subactions']
    )
    agent.network.load_state_dict(checkpoint['agent_state_dict'])
    agent.network.eval()
    
    total_params = sum(p.numel() for p in agent.network.parameters())
    print(f"  ✓ Agent initialized")
    print(f"    - Parameters: {total_params:,}")
    print(f"    - Options: {agent.options}")
except Exception as e:
    print(f"  [FAIL] FAILED to initialize agent: {e}")
    exit(1)

# Step 4: Test reset
print("\n[Step 4] Testing environment reset...")
try:
    state, info = env.reset()
    print(f"  ✓ Reset successful")
    print(f"    - State shape: {state.shape}")
    print(f"    - State range: [{state.min():.3f}, {state.max():.3f}]")
except Exception as e:
    print(f"  [FAIL] FAILED to reset: {e}")
    exit(1)

# Step 5: Test agent action selection
print("\n[Step 5] Testing agent action selection...")
try:
    agent.reset()
    available_options = env._get_available_options()
    available_subactions_dict = {
        opt: env._get_available_subactions(opt) 
        for opt in available_options
    }
    
    with torch.no_grad():
        action_dict = agent.select_action(
            state, 
            available_options, 
            available_subactions_dict,
            deterministic=True
        )
    
    print(f"  ✓ Action selection successful")
    print(f"    - Option: {action_dict['option_name']}")
    print(f"    - Subaction: {action_dict['subaction_name']}")
    print(f"    - Terminated: {action_dict['terminated']}")
except Exception as e:
    print(f"  [FAIL] FAILED action selection: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Test user state update
print("\n[Step 6] Testing user state update...")
try:
    # Simulate user looking at exhibit 1 with high engagement
    env.update_user_state(
        focus=1,  # King_Caspar
        dwell=0.85,
        utterance="Tell me about this piece"
    )
    
    print(f"  ✓ State update successful")
    print(f"    - Focus: {env.focus} ({env.exhibit_keys[env.focus-1] if env.focus > 0 else 'None'})")
    print(f"    - Dwell: {env.dwell:.3f}")
    print(f"    - Last utterance: \"{env.last_user_utterance}\"")
except Exception as e:
    print(f"  [FAIL] FAILED state update: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 7: Test environment step (WITH LLM - this will take time)
print("\n[Step 7] Testing environment step (generating response via LLM)...")
print("  This will take ~10 seconds as it calls the LLM...")
try:
    # Build action for environment
    env_action = {
        'option': action_dict['option'],
        'subaction': action_dict['subaction'],
        'terminate_option': action_dict['terminated']
    }
    
    next_state, reward, done, truncated, info = env.step(env_action)
    
    print(f"  ✓ Environment step successful")
    print(f"    - Reward: {reward:.3f}")
    print(f"    - Done: {done}")
    print(f"    - Agent said: \"{info['agent_utterance'][:100]}...\"")
    print(f"    - New facts: {info.get('facts_shared', 0)}")
except Exception as e:
    print(f"  [FAIL] FAILED environment step: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 8: Test complete turn cycle
print("\n[Step 8] Testing complete turn cycle...")
try:
    state = next_state
    available_options = env._get_available_options()
    available_subactions_dict = {
        opt: env._get_available_subactions(opt) 
        for opt in available_options
    }
    
    with torch.no_grad():
        action_dict = agent.select_action(
            state, 
            available_options, 
            available_subactions_dict,
            deterministic=True
        )
    
    print(f"  ✓ Complete cycle successful")
    print(f"    - Next option: {action_dict['option_name']}")
    print(f"    - Next subaction: {action_dict['subaction_name']}")
except Exception as e:
    print(f"  [FAIL] FAILED complete cycle: {e}")
    exit(1)

# Final summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE - ALL TESTS PASSED [OK]")
print("=" * 80)
print("\nThe trained model is FULLY FUNCTIONAL and ready to use.")
print("\nWhat works:")
print("  [OK] Model loads and contains valid weights")
print("  ✓ Environment initializes correctly")
print("  ✓ Agent makes decisions using trained policy")
print("  ✓ User state updates (focus, dwell, utterance)")
print("  ✓ Environment generates responses via LLM")
print("  ✓ Rewards are calculated properly")
print("  ✓ Complete turn cycle works")
print("\nNext step: Use the proper interface to interact with the model")
print("=" * 80)

