"""
Comprehensive Test Suite for Trained Museum Agent

This single file contains all tests and demos:
1. Model verification (loads correctly, basic functionality)
2. Network output inspection (see decision-making internals)
3. Demo conversation (5-turn example interaction)
4. Single turn test (quick validation)

Usage:
    python test_agent.py              # Run all tests
    python test_agent.py --verify     # Just verification
    python test_agent.py --network    # Just network outputs
    python test_agent.py --demo       # Just demo conversation
    python test_agent.py --quick      # Quick single-turn test
"""

import torch
import numpy as np
import sys
import argparse

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent
import torch.nn.functional as F


class AgentTester:
    """Comprehensive agent testing"""
    
    def __init__(self, model_path="models/trained_agent.pt"):
        self.model_path = model_path
        self.checkpoint = None
        self.env = None
        self.agent = None
        
    def load_model(self):
        """Load model and initialize components"""
        print("=" * 80)
        print("LOADING TRAINED MODEL")
        print("=" * 80)
        
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        
        print(f"Model: {self.checkpoint.get('timestamp', 'N/A')}")
        print(f"Training reward: {self.checkpoint.get('avg_reward', 0):.3f}")
        print(f"State dim: {self.checkpoint.get('state_dim', 'N/A')}")
        
        # Initialize environment
        self.env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")
        
        # Initialize agent
        self.agent = ActorCriticAgent(
            state_dim=self.checkpoint['state_dim'],
            options=self.checkpoint['options'],
            subactions=self.checkpoint['subactions']
        )
        self.agent.network.load_state_dict(self.checkpoint['agent_state_dict'])
        self.agent.network.eval()
        
        print(f"Parameters: {sum(p.numel() for p in self.agent.network.parameters()):,}")
        print("=" * 80)
        print()
        
    def test_verification(self):
        """Test 1: Basic model verification"""
        print("\n" + "=" * 80)
        print("TEST 1: MODEL VERIFICATION")
        print("=" * 80)
        
        tests_passed = 0
        tests_total = 7
        
        # Test 1: Model loads
        print("\n[1/7] Model loads...")
        try:
            assert self.checkpoint is not None
            print("    [PASS] Model loaded successfully")
            tests_passed += 1
        except:
            print("    [FAIL] Model failed to load")
            return False
        
        # Test 2: Environment initializes
        print("\n[2/7] Environment initializes...")
        try:
            assert self.env is not None
            assert len(self.env.exhibit_keys) == 8
            print(f"    [PASS] Environment ready with {len(self.env.exhibit_keys)} exhibits")
            tests_passed += 1
        except:
            print("    [FAIL] Environment initialization failed")
            return False
        
        # Test 3: Agent initializes
        print("\n[3/7] Agent initializes...")
        try:
            assert self.agent is not None
            print("    [PASS] Agent initialized")
            tests_passed += 1
        except:
            print("    [FAIL] Agent initialization failed")
            return False
        
        # Test 4: Reset works
        print("\n[4/7] Environment reset...")
        try:
            state, _ = self.env.reset()
            assert state.shape == (157,)
            print(f"    [PASS] Reset successful, state shape: {state.shape}")
            tests_passed += 1
        except Exception as e:
            print(f"    [FAIL] Reset failed: {e}")
            return False
        
        # Test 5: Agent action selection
        print("\n[5/7] Agent action selection...")
        try:
            self.agent.reset()
            available_options = self.env._get_available_options()
            available_subactions_dict = {
                opt: self.env._get_available_subactions(opt) 
                for opt in available_options
            }
            
            with torch.no_grad():
                action_dict = self.agent.select_action(
                    state, available_options, available_subactions_dict, deterministic=True
                )
            
            assert 'option' in action_dict
            assert 'subaction' in action_dict
            print(f"    [PASS] Selected: {action_dict['option_name']} / {action_dict['subaction_name']}")
            tests_passed += 1
        except Exception as e:
            print(f"    [FAIL] Action selection failed: {e}")
            return False
        
        # Test 6: State update
        print("\n[6/7] User state update...")
        try:
            self.env.update_user_state(
                focus=1,
                dwell=0.85,
                utterance="Tell me about this"
            )
            assert self.env.focus == 1
            assert self.env.dwell == 0.85
            print("    [PASS] State updated (focus=1, dwell=0.85)")
            tests_passed += 1
        except Exception as e:
            print(f"    [FAIL] State update failed: {e}")
            return False
        
        # Test 7: Environment step
        print("\n[7/7] Environment step (calling LLM, ~10 sec)...")
        try:
            env_action = {
                'option': action_dict['option'],
                'subaction': action_dict['subaction'],
                'terminate_option': action_dict['terminated']
            }
            
            next_state, reward, done, truncated, info = self.env.step(env_action)
            
            assert 'agent_utterance' in info
            assert len(info['agent_utterance']) > 0
            print(f"    [PASS] Step successful, reward: {reward:.3f}")
            print(f"    Response: \"{info['agent_utterance'][:80]}...\"")
            tests_passed += 1
        except Exception as e:
            print(f"    [FAIL] Environment step failed: {e}")
            return False
        
        # Summary
        print("\n" + "=" * 80)
        print(f"VERIFICATION COMPLETE: {tests_passed}/{tests_total} tests passed")
        print("=" * 80)
        
        return tests_passed == tests_total
    
    def test_network_outputs(self):
        """Test 2: Inspect neural network outputs"""
        print("\n" + "=" * 80)
        print("TEST 2: NEURAL NETWORK OUTPUT INSPECTION")
        print("=" * 80)
        
        # Setup scenario
        state, _ = self.env.reset()
        self.agent.reset()
        
        print("\n[Scenario]")
        print("Visitor: 'Tell me about this painting'")
        print("Focus: King_Caspar (1)")
        print("Dwell: 0.85 (high engagement)")
        
        self.env.update_user_state(
            focus=1,
            dwell=0.85,
            utterance="Tell me about this painting"
        )
        state = self.env._get_obs()
        
        print("\n[State Vector]")
        print(f"Dimension: {len(state)}")
        print(f"  Focus (0-8):     {state[0:9]}")
        print(f"  History (9-20):  {state[9:21].round(2)}")
        print(f"  Intent (21-84):  [{state[21]:.3f}, ..., {state[84]:.3f}] (64-d)")
        print(f"  Context (85-148): [{state[85]:.3f}, ..., {state[148]:.3f}] (64-d)")
        
        # Get network outputs
        print("\n[Neural Network Forward Pass]")
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            outputs = self.agent.network.forward(state_tensor)
        
        # Option logits and probabilities
        option_logits = outputs['option_logits'][0].numpy()
        option_probs = F.softmax(outputs['option_logits'][0], dim=-1).numpy()
        
        print("\nOption Probabilities:")
        for opt, prob in zip(self.agent.options, option_probs):
            bar = '█' * int(prob * 50)
            print(f"  {opt:20s}: {prob:6.1%} {bar}")
        
        # Subaction probabilities for top option
        top_option_idx = option_probs.argmax()
        top_option = self.agent.options[top_option_idx]
        explain_probs = F.softmax(outputs['intra_option_logits'][top_option_idx][0], dim=-1).numpy()
        
        print(f"\nSubaction Probabilities (for '{top_option}'):")
        subs = self.agent.subactions[top_option]
        for sub, prob in zip(subs, explain_probs[:len(subs)]):
            bar = '█' * int(prob * 50)
            print(f"  {sub:20s}: {prob:6.1%} {bar}")
        
        # Q-values
        q_values = outputs['option_values'][0].numpy()
        print("\nOption Q-Values (expected future reward):")
        for opt, qval in zip(self.agent.options, q_values):
            print(f"  {opt:20s}: {qval:7.3f}")
        
        print(f"\nState Value: {outputs['state_value'][0].item():.3f}")
        
        print("\n" + "=" * 80)
        return True
    
    def test_demo_conversation(self):
        """Test 3: Run 5-turn demo conversation"""
        print("\n" + "=" * 80)
        print("TEST 3: DEMO CONVERSATION (5 turns)")
        print("=" * 80)
        
        # Reset for clean start
        state, _ = self.env.reset()
        self.agent.reset()
        
        conversations = [
            ("Hello! This painting caught my eye.", 1, 0.82),
            ("That's interesting! Tell me more.", 1, 0.95),
            ("What other pieces do you have?", 1, 0.68),
            ("This necklace is beautiful!", 4, 0.91),
            ("What materials is it made from?", 4, 0.89),
        ]
        
        for i, (utterance, focus, dwell) in enumerate(conversations, 1):
            print(f"\n{'='*80}")
            print(f"TURN {i}")
            print(f"{'='*80}")
            
            exhibit = self.env.exhibit_keys[focus-1]
            print(f"Visitor: \"{utterance}\"")
            print(f"Looking at: [{focus}] {exhibit}")
            print(f"Dwell: {dwell:.2f}")
            
            # Update state
            self.env.update_user_state(focus=focus, dwell=dwell, utterance=utterance)
            state = self.env._get_obs()
            
            # Agent decision
            available_options = self.env._get_available_options()
            available_subactions_dict = {
                opt: self.env._get_available_subactions(opt) 
                for opt in available_options
            }
            
            with torch.no_grad():
                action_dict = self.agent.select_action(
                    state, available_options, available_subactions_dict, deterministic=True
                )
            
            print(f"\nAgent decision: {action_dict['option_name']} / {action_dict['subaction_name']}")
            
            # Execute
            env_action = {
                'option': action_dict['option'],
                'subaction': action_dict['subaction'],
                'terminate_option': action_dict['terminated']
            }
            
            try:
                next_state, reward, done, truncated, info = self.env.step(env_action)
                
                print(f"\nAgent: \"{info['agent_utterance']}\"")
                print(f"\nReward: {reward:.3f} (Engagement: {info['reward_engagement']:.3f}, Novelty: {info['reward_novelty']:.3f})")
                print(f"Facts shared: {info['facts_shared']}")
                
                state = next_state
            except Exception as e:
                print(f"\nError: {e}")
                break
        
        print(f"\n{'='*80}")
        print("CONVERSATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total turns: {self.env.turn_count}")
        print(f"Total reward: {self.env.session_reward:.3f}")
        print(f"Avg reward/turn: {self.env.session_reward/self.env.turn_count:.3f}")
        print(f"Facts shared: {sum(len(facts) for facts in self.env.facts_mentioned_per_exhibit.values())}")
        print(f"{'='*80}")
        
        return True
    
    def test_quick(self):
        """Test 4: Quick single-turn test"""
        print("\n" + "=" * 80)
        print("TEST 4: QUICK SINGLE-TURN TEST")
        print("=" * 80)
        
        state, _ = self.env.reset()
        self.agent.reset()
        
        print("\nInput: 'Hello!' at King_Caspar, dwell=0.8")
        
        self.env.update_user_state(focus=1, dwell=0.8, utterance="Hello!")
        state = self.env._get_obs()
        
        available_options = self.env._get_available_options()
        available_subactions_dict = {
            opt: self.env._get_available_subactions(opt) 
            for opt in available_options
        }
        
        with torch.no_grad():
            action_dict = self.agent.select_action(
                state, available_options, available_subactions_dict, deterministic=True
            )
        
        print(f"Decision: {action_dict['option_name']} / {action_dict['subaction_name']}")
        
        env_action = {
            'option': action_dict['option'],
            'subaction': action_dict['subaction'],
            'terminate_option': action_dict['terminated']
        }
        
        next_state, reward, done, truncated, info = self.env.step(env_action)
        
        print(f"Response: \"{info['agent_utterance'][:100]}...\"")
        print(f"Reward: {reward:.3f}")
        
        print("\n" + "=" * 80)
        return True


def main():
    parser = argparse.ArgumentParser(description='Test trained museum agent')
    parser.add_argument('--verify', action='store_true', help='Run verification tests only')
    parser.add_argument('--network', action='store_true', help='Show network outputs only')
    parser.add_argument('--demo', action='store_true', help='Run demo conversation only')
    parser.add_argument('--quick', action='store_true', help='Quick single-turn test')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    
    args = parser.parse_args()
    
    # Default to all if no flags specified
    if not (args.verify or args.network or args.demo or args.quick):
        args.all = True
    
    # Initialize tester
    tester = AgentTester()
    tester.load_model()
    
    results = {}
    
    # Run requested tests
    if args.verify or args.all:
        results['verification'] = tester.test_verification()
    
    if args.network or args.all:
        results['network'] = tester.test_network_outputs()
    
    if args.demo or args.all:
        results['demo'] = tester.test_demo_conversation()
    
    if args.quick:
        results['quick'] = tester.test_quick()
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

