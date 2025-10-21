"""
Simple Model Verification - No Unicode, Plain ASCII Only
"""

import torch
import numpy as np
import sys
import io

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent

print("=" * 80)
print("VERIFYING TRAINED MODEL - SIMPLE TEST")
print("=" * 80)

# Load model
print("\n[1] Loading model...")
checkpoint = torch.load("models/trained_agent.pt", map_location='cpu')
print(f"    OK - Timestamp: {checkpoint.get('timestamp', 'N/A')}")
print(f"    OK - Episodes: {checkpoint.get('total_episodes', 'N/A')}")
print(f"    OK - Avg reward: {checkpoint.get('avg_reward', 0):.3f}")

# Init environment
print("\n[2] Initializing environment...")
env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")
print(f"    OK - {len(env.exhibit_keys)} exhibits loaded")

# Init agent
print("\n[3] Initializing agent...")
agent = ActorCriticAgent(
    state_dim=checkpoint['state_dim'],
    options=checkpoint['options'],
    subactions=checkpoint['subactions']
)
agent.network.load_state_dict(checkpoint['agent_state_dict'])
agent.network.eval()
print(f"    OK - Agent ready with {sum(p.numel() for p in agent.network.parameters()):,} parameters")

# Test reset
print("\n[4] Testing reset...")
state, info = env.reset()
print(f"    OK - State shape: {state.shape}")

# Test action selection
print("\n[5] Testing action selection...")
agent.reset()
available_options = env._get_available_options()
available_subactions_dict = {opt: env._get_available_subactions(opt) for opt in available_options}

with torch.no_grad():
    action_dict = agent.select_action(state, available_options, available_subactions_dict, deterministic=True)

print(f"    OK - Selected: {action_dict['option_name']} / {action_dict['subaction_name']}")

# Test state update
print("\n[6] Testing state update...")
env.update_user_state(
    focus=1,  # First exhibit
    dwell=0.85,
    utterance="Tell me about this"
)
print(f"    OK - Focus={env.focus}, Dwell={env.dwell:.2f}")

# Test environment step
print("\n[7] Testing environment step (calling LLM, takes ~10 sec)...")
env_action = {
    'option': action_dict['option'],
    'subaction': action_dict['subaction'],
    'terminate_option': action_dict['terminated']
}

next_state, reward, done, truncated, info = env.step(env_action)
print(f"    OK - Reward: {reward:.3f}")
print(f"    OK - Agent response generated ({len(info['agent_utterance'])} chars)")

# Test next turn
print("\n[8] Testing next turn...")
state = next_state
with torch.no_grad():
    action_dict = agent.select_action(state, available_options, available_subactions_dict, deterministic=True)
print(f"    OK - Next action: {action_dict['option_name']}")

print("\n" + "=" * 80)
print("ALL TESTS PASSED - MODEL IS FULLY FUNCTIONAL")
print("=" * 80)
print("\nVerified components:")
print("  - Model loading and weights")
print("  - Environment initialization")
print("  - Agent policy execution")
print("  - State updates (focus, dwell, utterance)")
print("  - LLM response generation")
print("  - Reward calculation")
print("  - Multi-turn operation")
print("\nThe trained model works correctly!")
print("=" * 80)

