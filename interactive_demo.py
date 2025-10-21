"""
Interactive Demo - Chat with the Trained HRL Museum Agent

This script lets you interact with the trained museum guide agent.
You play the role of a visitor, and the agent guides you through the exhibits.
"""

import torch
import numpy as np
from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent
from src.utils.dialoguebert_intent_recognizer import DialogueBERTIntentRecognizer
import sys

class InteractiveMuseumDemo:
    def __init__(self, model_path="models/trained_agent.pt"):
        """Initialize interactive demo"""
        print("\n" + "=" * 80)
        print("           INTERACTIVE MUSEUM GUIDE - HRL AGENT DEMO")
        print("=" * 80)
        print("\nLoading trained agent...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize environment
        self.env = MuseumDialogueEnv(
            knowledge_graph_path="museum_knowledge_graph.json"
        )
        
        # Initialize agent
        self.agent = ActorCriticAgent(
            state_dim=checkpoint['state_dim'],
            options=checkpoint['options'],
            subactions=checkpoint['subactions']
        )
        
        # Load weights
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.network.eval()
        
        # Initialize intent recognizer for your messages
        self.intent_recognizer = DialogueBERTIntentRecognizer()
        
        print(f"\n[OK] Agent loaded successfully!")
        print(f"     Training: {checkpoint.get('timestamp', 'N/A')}")
        print(f"     Avg Reward: {checkpoint.get('avg_reward', 0):.3f}")
        print(f"     Parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}")
        
    def run_interactive_session(self):
        """Run interactive conversation"""
        print("\n" + "=" * 80)
        print("                        WELCOME TO THE MUSEUM!")
        print("=" * 80)
        print("\nYou are now chatting with an AI museum guide.")
        print("The guide will help you explore various exhibits.")
        print("\nAvailable exhibits:")
        for i, exhibit in enumerate(self.env.exhibit_keys, 1):
            print(f"  {i}. {exhibit.replace('_', ' ')}")
        
        print("\nCommands:")
        print("  - Type your messages naturally")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("  - Type 'help' for more options")
        print("\n" + "=" * 80)
        
        # Reset environment
        state, _ = self.env.reset()
        
        # Start conversation with agent's greeting
        print("\n[Museum Guide]: Hello! Welcome to our museum. I'm here to guide you")
        print("                through our fascinating exhibits. What would you like")
        print("                to explore today?")
        
        turn = 0
        max_turns = 30
        
        while turn < max_turns:
            turn += 1
            
            # Get user input
            print(f"\n[You] (Turn {turn}): ", end="")
            sys.stdout.flush()
            user_input = input().strip()
            
            if not user_input:
                print("        (Please type something)")
                turn -= 1
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n[Museum Guide]: Thank you for visiting! Come back soon!")
                break
            
            if user_input.lower() == 'help':
                self._show_help()
                turn -= 1
                continue
            
            if user_input.lower() == 'status':
                self._show_status()
                turn -= 1
                continue
            
            # Process user input through DialogueBERT
            try:
                intent, intent_embedding = self.intent_recognizer.recognize_intent(user_input)
                
                # Update environment with user response
                # (This is a simplified version - in full implementation, 
                # the environment would process this through its step function)
                self.env.last_user_utterance = user_input
                self.env._last_user_intent = intent
                
                # Update DialogueBERT embeddings in state
                # Project to 64-d for state representation
                state_intent_64d = self.intent_recognizer.project_to_state_dim(intent_embedding)
                
                # Get dialogue context
                context_embedding = self.intent_recognizer.get_dialogue_context(
                    user_utterance=user_input,
                    agent_utterance=self.env.dialogue_history[-1] if self.env.dialogue_history else ""
                )
                state_context_64d = self.intent_recognizer.project_to_state_dim(context_embedding)
                
                # Update state with new embeddings
                # (State indices: 0-8: focus, 9-20: history, 21-84: intent, 85-148: context)
                state[21:85] = state_intent_64d
                state[85:149] = state_context_64d
                
            except Exception as e:
                print(f"        (Intent recognition skipped: {e})")
                intent = "statement"
            
            # Get available options and subactions
            available_options = self.env._get_available_options()
            available_subactions_dict = {opt: self.env._get_available_subactions(opt) 
                                        for opt in available_options}
            
            # Agent selects action
            with torch.no_grad():
                action_dict = self.agent.select_action(
                    state, available_options, available_subactions_dict,
                    deterministic=True  # Use greedy policy for demo
                )
            
            option_idx = action_dict['option']
            subaction_idx = action_dict['subaction']
            
            option_name = available_options[option_idx]
            subaction_name = available_subactions_dict[option_name][subaction_idx]
            
            # Generate agent response
            try:
                from src.utils.dialogue_planner import DialoguePlanner
                from LLM_CONFIG import get_agent_llm
                
                planner = DialoguePlanner(
                    knowledge_graph=self.env.knowledge_graph,
                    llm_handler=get_agent_llm()
                )
                
                # Get current exhibit
                current_exhibit = self.env.exhibit_keys[self.env.focus - 1] if self.env.focus > 0 else None
                
                # Generate response
                agent_response, new_facts = planner.generate_response(
                    option=option_name,
                    subaction=subaction_name,
                    current_exhibit=current_exhibit,
                    user_utterance=user_input,
                    dialogue_history=self.env.dialogue_history[-4:],  # Last 4 turns
                    facts_mentioned=self.env.facts_mentioned_per_exhibit,
                    available_exhibits=self.env.exhibit_keys
                )
                
                # Update dialogue history
                self.env.dialogue_history.append(user_input)
                self.env.dialogue_history.append(agent_response)
                
                # Keep only recent history
                if len(self.env.dialogue_history) > 6:
                    self.env.dialogue_history = self.env.dialogue_history[-6:]
                
                # Display agent response
                print(f"\n[Museum Guide]: {agent_response}")
                
                # Show what action was chosen (for debugging)
                if False:  # Set to True to see agent's internal decisions
                    print(f"\n        (Agent chose: {option_name}/{subaction_name})")
                
            except Exception as e:
                print(f"\n[Museum Guide]: I apologize, I'm having trouble formulating a response.")
                print(f"        (Error: {e})")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("                  Thank you for visiting the museum!")
        print("=" * 80)
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "-" * 80)
        print("HELP - Available Commands:")
        print("-" * 80)
        print("  help   - Show this help message")
        print("  status - Show current conversation status")
        print("  quit   - End the conversation")
        print("\nTips for natural conversation:")
        print("  - Ask questions about exhibits")
        print("  - Express interest or curiosity")
        print("  - Request to see different exhibits")
        print("  - The agent will adapt to your interests!")
        print("-" * 80)
    
    def _show_status(self):
        """Show conversation status"""
        print("\n" + "-" * 80)
        print("CONVERSATION STATUS:")
        print("-" * 80)
        current_exhibit = self.env.exhibit_keys[self.env.focus - 1] if self.env.focus > 0 else "None"
        print(f"  Current Exhibit: {current_exhibit}")
        print(f"  Turns: {len(self.env.dialogue_history) // 2}")
        
        # Count facts shared
        total_facts = sum(len(facts) for facts in self.env.facts_mentioned_per_exhibit.values())
        print(f"  Facts Shared: {total_facts}")
        
        # Show visited exhibits
        visited = [ex for ex, count in enumerate(self.env.explained) if count > 0]
        print(f"  Exhibits Visited: {len(visited)}")
        print("-" * 80)


def main():
    """Main entry point"""
    try:
        demo = InteractiveMuseumDemo("models/trained_agent.pt")
        demo.run_interactive_session()
    except KeyboardInterrupt:
        print("\n\nConversation interrupted. Goodbye!")
    except FileNotFoundError:
        print("\nError: Trained model not found!")
        print("Please run: python train.py --episodes 1 --turns 15")
        print("to train the model first.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

