"""
Show Raw Network Outputs

This demonstrates what the neural network actually produces
when making a decision.
"""

import torch
import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent
import torch.nn.functional as F

# Load model
print("="*80)
print("NEURAL NETWORK DECISION ANALYSIS")
print("="*80)

checkpoint = torch.load("models/trained_agent.pt", map_location='cpu')
env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")

agent = ActorCriticAgent(
    state_dim=checkpoint['state_dim'],
    options=checkpoint['options'],
    subactions=checkpoint['subactions']
)
agent.network.load_state_dict(checkpoint['agent_state_dict'])
agent.network.eval()

# Reset
state, _ = env.reset()
agent.reset()

# Simulate user input
print("\n[SCENARIO]")
print("Visitor says: 'Tell me about this painting'")
print("Looking at: King_Caspar (focus=1)")
print("Engagement: 0.85 (high)")
print()

env.update_user_state(
    focus=1,
    dwell=0.85,
    utterance="Tell me about this painting"
)
state = env._get_obs()

print("="*80)
print("STATE VECTOR BREAKDOWN")
print("="*80)
print(f"Total dimension: {len(state)}")
print(f"\nComponents:")
print(f"  Focus (0-8):     {state[0:9]}")
print(f"  History (9-20):  {state[9:21].round(2)}")
print(f"  Intent (21-84):  [{state[21]:.3f}, {state[22]:.3f}, ..., {state[84]:.3f}] (64-d)")
print(f"  Context (85-148): [{state[85]:.3f}, {state[86]:.3f}, ..., {state[148]:.3f}] (64-d)")

# Get network outputs
print("\n" + "="*80)
print("NEURAL NETWORK FORWARD PASS")
print("="*80)

state_tensor = torch.FloatTensor(state).unsqueeze(0)
with torch.no_grad():
    outputs = agent.network.forward(state_tensor)

print("\n[RAW NETWORK OUTPUTS]")
print("-"*80)

# Option logits
option_logits = outputs['option_logits'][0].numpy()
print("\n1. OPTION LOGITS (raw scores from network):")
for i, (opt, logit) in enumerate(zip(agent.options, option_logits)):
    print(f"   {opt:20s}: {logit:7.3f}")

# Convert to probabilities
option_probs = F.softmax(outputs['option_logits'][0], dim=-1).numpy()
print("\n2. OPTION PROBABILITIES (after softmax):")
for i, (opt, prob) in enumerate(zip(agent.options, option_probs)):
    bar = '█' * int(prob * 50)
    print(f"   {opt:20s}: {prob:6.1%} {bar}")
print(f"   {'':20s}   ↑")
print(f"   {'':20s}   Agent will choose highest: {agent.options[option_probs.argmax()]}")

# Intra-option logits for Explain
print("\n3. SUBACTION LOGITS (for 'Explain' option):")
explain_logits = outputs['intra_option_logits'][0][0].numpy()
explain_subs = agent.subactions['Explain']
for sub, logit in zip(explain_subs, explain_logits[:len(explain_subs)]):
    print(f"   {sub:20s}: {logit:7.3f}")

# Convert to probabilities
explain_probs = F.softmax(outputs['intra_option_logits'][0][0], dim=-1).numpy()
print("\n4. SUBACTION PROBABILITIES (for 'Explain'):")
for sub, prob in zip(explain_subs, explain_probs[:len(explain_subs)]):
    bar = '█' * int(prob * 50)
    print(f"   {sub:20s}: {prob:6.1%} {bar}")
print(f"   {'':20s}   ↑")
print(f"   {'':20s}   Agent will choose: {explain_subs[explain_probs[:len(explain_subs)].argmax()]}")

# Termination probabilities
print("\n5. TERMINATION PROBABILITIES (should we switch options?):")
term_probs = outputs['termination_probs'][0].numpy()
for opt, prob in zip(agent.options, term_probs):
    print(f"   {opt:20s}: {prob:6.1%} (threshold: 50%)")

# Q-values (critic outputs)
print("\n6. OPTION Q-VALUES (expected future reward per option):")
q_values = outputs['option_values'][0].numpy()
for opt, qval in zip(agent.options, q_values):
    print(f"   {opt:20s}: {qval:7.3f}")

print(f"\n7. STATE VALUE: {outputs['state_value'][0].item():.3f}")
print("   (Expected total future reward from this state)")

# Final decision
print("\n" + "="*80)
print("AGENT'S FINAL DECISION")
print("="*80)

available_options = env._get_available_options()
available_subactions_dict = {opt: env._get_available_subactions(opt) for opt in available_options}

with torch.no_grad():
    action_dict = agent.select_action(state, available_options, available_subactions_dict, deterministic=True)

print(f"\nOption:    {action_dict['option_name']}")
print(f"Subaction: {action_dict['subaction_name']}")
print(f"Terminate: {action_dict['terminated']}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nWhy did the network choose 'Explain / ExplainNewFact'?")
print()
print("1. HIGH OPTION PROBABILITY:")
print(f"   'Explain' had {option_probs[0]*100:.1f}% probability")
print(f"   The network learned: high dwell + new exhibit = explain")
print()
print("2. HIGH SUBACTION PROBABILITY:")
print(f"   'ExplainNewFact' had {explain_probs[0]*100:.1f}% probability")
print(f"   The network learned: no facts shared yet = introduce new fact")
print()
print("3. LOW TERMINATION PROBABILITY:")
print(f"   Only {term_probs[0]*100:.1f}% chance to terminate")
print(f"   The network learned: just starting = don't switch strategy yet")
print()
print("4. THESE WEIGHTS WERE LEARNED:")
print("   During training, the agent got high rewards for explaining")
print("   when visitors showed high engagement (dwell=0.85).")
print("   The network updated its weights to prefer this action.")
print()
print("="*80)

