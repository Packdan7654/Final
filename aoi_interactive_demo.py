"""
AOI-Based Interactive Museum Demo

This is how the system ACTUALLY works:
- You select which exhibit/AOI you're looking at
- The system updates the state based on your gaze
- The agent responds to what you're looking at
- You can type messages too, but AOI focus is primary

This matches the training setup exactly.
"""

import torch
import numpy as np
from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent
from src.simulator.sim8_adapter import Sim8Simulator
import sys
import time

class AOIInteractiveDemo:
    def __init__(self, model_path="models/trained_agent.pt"):
        """Initialize AOI-based interactive demo"""
        print("\n" + "=" * 80)
        print("        AOI-BASED MUSEUM GUIDE - INTERACTIVE DEMO")
        print("=" * 80)
        print("\nLoading system components...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize environment (this handles all the dialogue generation)
        self.env = MuseumDialogueEnv(
            knowledge_graph_path="museum_knowledge_graph.json"
        )
        
        # Initialize simulator (for response generation)
        self.simulator = Sim8Simulator(exhibits=self.env.exhibit_keys)
        
        # Initialize agent
        self.agent = ActorCriticAgent(
            state_dim=checkpoint['state_dim'],
            options=checkpoint['options'],
            subactions=checkpoint['subactions']
        )
        
        # Load weights
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.network.eval()
        
        print(f"\n[OK] System loaded successfully!")
        print(f"     Model: {checkpoint.get('timestamp', 'N/A')}")
        print(f"     Avg Reward: {checkpoint.get('avg_reward', 0):.3f}")
        print(f"     Parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}")
        
    def run_interactive_session(self):
        """Run AOI-based interactive conversation"""
        print("\n" + "=" * 80)
        print("                 WELCOME TO THE INTERACTIVE MUSEUM!")
        print("=" * 80)
        print("\nHow this works:")
        print("  1. You look at exhibits (by selecting an AOI number)")
        print("  2. The system detects what you're looking at")
        print("  3. The AI guide responds based on your gaze and interest")
        print("  4. You can also type messages to the guide")
        print("\nThis simulates real museum experience with eye-tracking!")
        
        # Show available exhibits/AOIs
        print("\n" + "-" * 80)
        print("AVAILABLE EXHIBITS (AOIs):")
        print("-" * 80)
        for i, exhibit in enumerate(self.env.exhibit_keys, 1):
            print(f"  [{i}] {exhibit.replace('_', ' ')}")
        print(f"  [0] Look away / No focus")
        print("-" * 80)
        
        # Reset environment and simulator
        state, _ = self.env.reset()
        self.simulator.initialize_session(persona="Agreeable")
        self.agent.reset()
        
        # Get starting AOI from simulator
        current_aoi = self.simulator.get_current_aoi()
        if current_aoi in self.env.exhibit_keys:
            focus_idx = self.env.exhibit_keys.index(current_aoi) + 1
        else:
            focus_idx = 0
        
        print(f"\nYou start by looking at: {current_aoi}")
        print("\nCommands: 'help', 'status', 'quit'")
        print("=" * 80)
        
        turn = 0
        max_turns = 30
        
        while turn < max_turns:
            turn += 1
            print(f"\n{'=' * 80}")
            print(f"TURN {turn}")
            print(f"{'=' * 80}")
            
            # Show current state
            current_exhibit = self.env.exhibit_keys[focus_idx - 1] if focus_idx > 0 else "Nothing"
            print(f"Currently looking at: [{focus_idx}] {current_exhibit}")
            
            # Get user input for AOI selection
            print(f"\n[Action] What do you want to do?")
            print(f"  - Type a number (0-{len(self.env.exhibit_keys)}) to look at an exhibit")
            print(f"  - Type a message to talk to the guide")
            print(f"  - Type 'stay' to keep looking at current exhibit")
            print(f"  - Type 'quit' to exit")
            
            user_input = input(f"\n> ").strip()
            
            if not user_input:
                print("(Staying at current exhibit)")
                user_input = "stay"
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n[Museum Guide]: Thank you for visiting! Goodbye!")
                break
            
            if user_input.lower() == 'help':
                self._show_help()
                turn -= 1
                continue
            
            if user_input.lower() == 'status':
                self._show_status()
                turn -= 1
                continue
            
            # Check if user is selecting an AOI
            user_message = ""
            if user_input.isdigit():
                new_focus = int(user_input)
                if 0 <= new_focus <= len(self.env.exhibit_keys):
                    focus_idx = new_focus
                    if focus_idx == 0:
                        print(f"\n[You look away from exhibits]")
                        user_message = "I'm just looking around."
                    else:
                        exhibit_name = self.env.exhibit_keys[focus_idx - 1]
                        print(f"\n[You look at: {exhibit_name}]")
                        user_message = f"I'm looking at the {exhibit_name.replace('_', ' ')}."
                        # Update simulator AOI
                        self.simulator.current_aoi = exhibit_name
                else:
                    print("Invalid exhibit number!")
                    turn -= 1
                    continue
            elif user_input.lower() == 'stay':
                user_message = "Tell me more about this."
            else:
                # User typed a message
                user_message = user_input
            
            # Update environment with current focus
            self.env.update_user_state(
                focus=focus_idx,
                utterance=user_message
            )
            
            # Get dwell time from simulator
            gaze_data = self.simulator.simulate_gaze()
            dwell_time = gaze_data['dwell_time']
            self.env.dwell = dwell_time
            
            # Get current observation
            state = self.env._get_obs()
            
            # Agent selects action
            available_options = self.env._get_available_options()
            available_subactions_dict = {
                opt: self.env._get_available_subactions(opt) 
                for opt in available_options
            }
            
            with torch.no_grad():
                action_dict = self.agent.select_action(
                    state, available_options, available_subactions_dict,
                    deterministic=True
                )
            
            # Convert to environment action format
            env_action = {
                'option': action_dict['option'],
                'subaction': action_dict['subaction'],
                'terminate_option': action_dict['terminated']
            }
            
            print(f"\n[Agent Decision]")
            print(f"  Option: {action_dict['option_name']}")
            print(f"  Subaction: {action_dict['subaction_name']}")
            print(f"  Dwell Time: {dwell_time:.3f}")
            
            # Execute step in environment (this generates agent response via LLM)
            try:
                next_state, reward, done, truncated, info = self.env.step(env_action)
                
                # Display agent's response
                agent_utterance = info.get('agent_utterance', '')
                print(f"\n[Museum Guide]:")
                print(f"  {agent_utterance}")
                
                # Get simulator response to agent
                simulator_response = self.simulator.generate_response(
                    agent_utterance=agent_utterance,
                    response_type=None  # Let simulator decide
                )
                
                user_response = simulator_response['user_utterance']
                print(f"\n[Visitor Response]:")
                print(f"  {user_response}")
                
                # Show reward
                print(f"\n[Metrics]")
                print(f"  Reward: {reward:.3f}")
                print(f"  User Intent: {info.get('user_intent', 'N/A')}")
                print(f"  New Facts: {info.get('new_facts_count', 0)}")
                
                # Update state
                state = next_state
                
                # Check if done
                if done:
                    print(f"\n[Episode Complete]")
                    break
                    
            except Exception as e:
                print(f"\n[ERROR]: {e}")
                import traceback
                traceback.print_exc()
                print("\nContinuing anyway...")
        
        # Final summary
        print("\n" + "=" * 80)
        print("                    SESSION COMPLETE")
        print("=" * 80)
        self._show_status()
        print("=" * 80)
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "-" * 80)
        print("HELP - How to Use:")
        print("-" * 80)
        print("\nAOI Selection (Gaze):")
        print("  - Type 0-8 to look at different exhibits")
        print("  - Type 'stay' to keep looking at current exhibit")
        print(f"  - The agent responds to WHERE you're looking")
        print("\nText Messages:")
        print("  - Type any text to send a message to the guide")
        print("  - Messages are combined with your gaze focus")
        print("\nCommands:")
        print("  - 'help'   - Show this help")
        print("  - 'status' - Show conversation statistics")
        print("  - 'quit'   - End the session")
        print("\nThis simulates eye-tracking in a real museum!")
        print("-" * 80)
    
    def _show_status(self):
        """Show conversation status"""
        print("\n" + "-" * 80)
        print("SESSION STATUS:")
        print("-" * 80)
        current_exhibit = self.env.exhibit_keys[self.env.focus - 1] if self.env.focus > 0 else "None"
        print(f"  Current Focus: [{self.env.focus}] {current_exhibit}")
        print(f"  Current Dwell: {self.env.dwell:.3f}")
        print(f"  Turns: {len(self.env.dialogue_history) // 2}")
        
        # Count facts shared
        total_facts = sum(len(facts) for facts in self.env.facts_mentioned_per_exhibit.values())
        print(f"  Facts Shared: {total_facts}")
        
        # Show which exhibits have been explained
        explained_exhibits = [self.env.exhibit_keys[i] for i, count in enumerate(self.env.explained) if count > 0]
        print(f"  Exhibits Visited: {len(explained_exhibits)}")
        if explained_exhibits:
            print(f"    {', '.join(explained_exhibits)}")
        
        # Show option usage
        print(f"  Actions Used: {dict(self.env.actions_used)}")
        print("-" * 80)


def main():
    """Main entry point"""
    try:
        demo = AOIInteractiveDemo("models/trained_agent.pt")
        demo.run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Goodbye!")
    except FileNotFoundError:
        print("\nError: Trained model not found!")
        print("Please run: python train.py --episodes 1 --turns 15")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

