"""
Proper Museum Agent Interface

Input: dialogue (utterance), focus (exhibit index), dwell time
Output: Agent's response based on trained policy

This is the CORRECT way to interact with the system as it was designed.
"""

import torch
import numpy as np
import sys

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent

class MuseumAgentInterface:
    def __init__(self, model_path="models/trained_agent.pt"):
        """Initialize the interface with trained model"""
        print("=" * 80)
        print("LOADING MUSEUM AGENT")
        print("=" * 80)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Model: {checkpoint.get('timestamp', 'N/A')}")
        print(f"Training reward: {checkpoint.get('avg_reward', 0):.3f}")
        
        # Initialize environment
        self.env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")
        
        # Initialize agent
        self.agent = ActorCriticAgent(
            state_dim=checkpoint['state_dim'],
            options=checkpoint['options'],
            subactions=checkpoint['subactions']
        )
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.network.eval()
        
        print(f"Parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}")
        print("=" * 80)
        
        # Reset
        self.state, _ = self.env.reset()
        self.agent.reset()
        
        # Show exhibits
        print("\nAVAILABLE EXHIBITS:")
        for i, exhibit in enumerate(self.env.exhibit_keys, 1):
            print(f"  [{i}] {exhibit.replace('_', ' ')}")
        print("  [0] No focus / looking away")
        print()
    
    def process_turn(self, utterance: str, focus: int, dwell: float):
        """
        Process one turn of interaction
        
        Args:
            utterance: What the visitor says
            focus: Which exhibit they're looking at (0-8, where 0=no focus)
            dwell: How engaged they are (0.0-1.0)
        
        Returns:
            agent_response: What the agent says back
            info: Dictionary with turn info (reward, option used, etc.)
        """
        print("-" * 80)
        print(f"TURN {self.env.turn_count + 1}")
        print("-" * 80)
        
        # Validate inputs
        if not 0 <= focus <= len(self.env.exhibit_keys):
            raise ValueError(f"Focus must be 0-{len(self.env.exhibit_keys)}")
        if not 0.0 <= dwell <= 1.0:
            raise ValueError("Dwell must be between 0.0 and 1.0")
        
        # Show inputs
        exhibit_name = self.env.exhibit_keys[focus - 1] if focus > 0 else "None"
        print(f"Visitor utterance: \"{utterance}\"")
        print(f"Focus: [{focus}] {exhibit_name}")
        print(f"Dwell time: {dwell:.3f}")
        print()
        
        # Update environment state
        self.env.update_user_state(
            focus=focus,
            dwell=dwell,
            utterance=utterance
        )
        
        # Get current state
        self.state = self.env._get_obs()
        
        # Agent selects action
        available_options = self.env._get_available_options()
        available_subactions_dict = {
            opt: self.env._get_available_subactions(opt) 
            for opt in available_options
        }
        
        with torch.no_grad():
            action_dict = self.agent.select_action(
                self.state,
                available_options,
                available_subactions_dict,
                deterministic=True
            )
        
        print(f"Agent decision: {action_dict['option_name']} / {action_dict['subaction_name']}")
        print()
        
        # Execute environment step (generates LLM response)
        env_action = {
            'option': action_dict['option'],
            'subaction': action_dict['subaction'],
            'terminate_option': action_dict['terminated']
        }
        
        print("Generating response (LLM call, ~10 sec)...")
        next_state, reward, done, truncated, info = self.env.step(env_action)
        
        # Update state
        self.state = next_state
        
        # Display results
        print()
        print("=" * 80)
        print(f"AGENT RESPONSE:")
        print("=" * 80)
        print(info['agent_utterance'])
        print()
        print("-" * 80)
        print(f"Reward: {reward:.3f} (Engagement: {info['reward_engagement']:.3f}, Novelty: {info['reward_novelty']:.3f})")
        print(f"Facts shared: {info['facts_shared']}")
        if info.get('facts_mentioned_in_utterance'):
            print(f"Fact IDs: {info['facts_mentioned_in_utterance']}")
        print(f"Total turns: {self.env.turn_count}")
        print(f"Session reward: {self.env.session_reward:.3f}")
        print("=" * 80)
        print()
        
        return info['agent_utterance'], info


def main():
    """Run interactive demo"""
    interface = MuseumAgentInterface()
    
    print("=" * 80)
    print("MUSEUM AGENT INTERFACE - Ready for interaction")
    print("=" * 80)
    print()
    print("How to use:")
    print("  1. You provide: dialogue + focus (0-8) + dwell (0.0-1.0)")
    print("  2. Agent responds using trained policy")
    print()
    print("Commands:")
    print("  - Just press Enter to see an example")
    print("  - Type 'quit' to exit")
    print("  - Type 'status' to see current state")
    print("=" * 80)
    print()
    
    # Example turns
    examples = [
        ("Hello! I'm interested in historical artwork.", 1, 0.75),
        ("Tell me more about this piece.", 1, 0.92),
        ("What else do you have?", 0, 0.45),
        ("I'd like to see the necklace.", 4, 0.88),
    ]
    
    example_idx = 0
    
    while True:
        if example_idx < len(examples):
            print(f"\n[Example {example_idx + 1}/{len(examples)}] Press Enter to run, or type your own input:")
            utterance, focus, dwell = examples[example_idx]
            print(f"  Utterance: \"{utterance}\"")
            print(f"  Focus: {focus}")
            print(f"  Dwell: {dwell}")
            print()
        else:
            print("\nAll examples done. Provide your own input or type 'quit'.")
        
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'status':
            print(f"\nCurrent state:")
            print(f"  Turns: {interface.env.turn_count}")
            print(f"  Session reward: {interface.env.session_reward:.3f}")
            print(f"  Facts shared: {sum(len(facts) for facts in interface.env.facts_mentioned_per_exhibit.values())}")
            continue
        
        # Use example or custom input
        if not user_input and example_idx < len(examples):
            utterance, focus, dwell = examples[example_idx]
            example_idx += 1
        else:
            # Parse custom input
            parts = user_input.split('|')
            if len(parts) == 3:
                try:
                    utterance = parts[0].strip()
                    focus = int(parts[1].strip())
                    dwell = float(parts[2].strip())
                except:
                    print("Error: Format should be: utterance | focus | dwell")
                    print("Example: Tell me about this | 1 | 0.85")
                    continue
            else:
                print("To provide custom input, use format:")
                print("  utterance | focus | dwell")
                print("Example: Tell me about this | 1 | 0.85")
                continue
        
        # Process turn
        try:
            response, info = interface.process_turn(utterance, focus, dwell)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

