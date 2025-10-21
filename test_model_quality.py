"""
Test and Quality Assessment for Trained HRL Museum Agent

This script:
1. Loads the trained model
2. Runs test episodes with different visitor personas
3. Evaluates model quality across multiple dimensions
4. Provides detailed assessment with metrics
"""

import torch
import numpy as np
from src.environment.env import MuseumDialogueEnv
from src.agent.actor_critic_agent import ActorCriticAgent
from src.simulator.sim8_adapter import Sim8Simulator
import json
from datetime import datetime


class ModelQualityAssessment:
    def __init__(self, model_path="models/trained_agent.pt"):
        """Initialize quality assessment"""
        self.model_path = model_path
        self.results = []
        
    def load_model(self):
        """Load trained model and inspect checkpoint"""
        print("=" * 80)
        print("LOADING TRAINED MODEL")
        print("=" * 80)
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        print(f"[OK] Model loaded from: {self.model_path}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        print(f"  Training Episodes: {checkpoint.get('total_episodes', 'N/A')}")
        print(f"  Avg Training Reward: {checkpoint.get('avg_reward', 0):.3f}")
        print(f"  State Dimension: {checkpoint.get('state_dim', 'N/A')}")
        print(f"  Configuration: {checkpoint.get('config', {})}")
        print()
        
        # Initialize environment
        env = MuseumDialogueEnv(
            knowledge_graph_path="museum_knowledge_graph.json"
        )
        
        # Initialize simulator
        simulator = Sim8Simulator(exhibits=env.exhibit_keys)
        
        # Initialize agent
        agent = ActorCriticAgent(
            state_dim=checkpoint['state_dim'],
            options=checkpoint['options'],
            subactions=checkpoint['subactions']
        )
        
        # Load weights
        agent.network.load_state_dict(checkpoint['agent_state_dict'])
        agent.network.eval()  # Set to evaluation mode
        
        print(f"[OK] Agent initialized with {sum(p.numel() for p in agent.network.parameters())} parameters")
        print(f"  Options: {checkpoint['options']}")
        print(f"  Subactions per option: {len(checkpoint['subactions'][checkpoint['options'][0]])}")
        print(f"[OK] Simulator initialized with {len(simulator.personas)} personas")
        print()
        
        return env, agent, simulator, checkpoint
    
    def run_test_episode(self, env, agent, simulator, persona="Agreeable", max_turns=15, verbose=True):
        """Run a single test episode"""
        if verbose:
            print("=" * 80)
            print(f"TEST EPISODE - Persona: {persona}")
            print("=" * 80)
        
        # Reset environment
        state, info = env.reset()
        
        # Reset simulator with specific persona
        sim_state = simulator.reset(persona=persona)
        
        episode_data = {
            'persona': persona,
            'turns': [],
            'facts_shared': 0,
            'exhibits_covered': 0,
            'total_reward': 0.0,
            'engagement_scores': [],
            'option_usage': {},
            'subaction_usage': {}
        }
        
        done = False
        turn = 0
        
        while not done and turn < max_turns:
            turn += 1
            
            # Agent selects action
            with torch.no_grad():
                option_idx, subaction_idx, terminate = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step({
                'option': option_idx,
                'subaction': subaction_idx,
                'terminate': terminate
            })
            
            # Record turn data
            turn_data = {
                'turn': turn,
                'option': env.options[option_idx],
                'subaction': env.subactions[env.options[option_idx]][subaction_idx],
                'terminate': terminate,
                'reward': reward,
                'agent_utterance': info.get('agent_utterance', ''),
                'user_utterance': info.get('user_response', ''),
                'user_intent': info.get('user_intent', ''),
                'dwell_time': info.get('dwell_time', 0),
                'current_exhibit': info.get('current_exhibit', ''),
                'facts_this_turn': info.get('new_facts_count', 0)
            }
            
            episode_data['turns'].append(turn_data)
            episode_data['total_reward'] += reward
            episode_data['facts_shared'] += turn_data['facts_this_turn']
            episode_data['engagement_scores'].append(info.get('dwell_time', 0))
            
            # Count option/subaction usage
            option_name = env.options[option_idx]
            subaction_name = env.subactions[option_name][subaction_idx]
            episode_data['option_usage'][option_name] = episode_data['option_usage'].get(option_name, 0) + 1
            episode_data['subaction_usage'][subaction_name] = episode_data['subaction_usage'].get(subaction_name, 0) + 1
            
            if verbose:
                print(f"\n--- Turn {turn} ---")
                print(f"Option: {option_name} | Subaction: {subaction_name} | Terminate: {terminate}")
                print(f"[Agent]: {info.get('agent_utterance', '')[:100]}...")
                print(f"[User]:  {info.get('user_response', '')[:100]}...")
                print(f"Reward: {reward:.3f} | Intent: {info.get('user_intent', '')} | Dwell: {info.get('dwell_time', 0):.3f}")
            
            state = next_state
        
        # Count unique exhibits
        exhibits_visited = set([t['current_exhibit'] for t in episode_data['turns']])
        episode_data['exhibits_covered'] = len(exhibits_visited)
        
        if verbose:
            print("\n" + "=" * 80)
            print("EPISODE SUMMARY")
            print("=" * 80)
            print(f"Turns: {turn}")
            print(f"Total Reward: {episode_data['total_reward']:.3f}")
            print(f"Avg Reward/Turn: {episode_data['total_reward']/turn:.3f}")
            print(f"Facts Shared: {episode_data['facts_shared']}")
            print(f"Exhibits Covered: {episode_data['exhibits_covered']}")
            print(f"Avg Engagement: {np.mean(episode_data['engagement_scores']):.3f}")
            print(f"Option Usage: {episode_data['option_usage']}")
            print(f"Subaction Usage: {episode_data['subaction_usage']}")
            print("=" * 80)
        
        return episode_data
    
    def compute_quality_metrics(self, episodes_data):
        """Compute comprehensive quality metrics"""
        print("\n" + "=" * 80)
        print("QUALITY ASSESSMENT METRICS")
        print("=" * 80)
        
        # Aggregate metrics
        total_turns = sum(len(ep['turns']) for ep in episodes_data)
        avg_reward = np.mean([ep['total_reward'] for ep in episodes_data])
        avg_reward_per_turn = np.mean([ep['total_reward']/len(ep['turns']) for ep in episodes_data])
        avg_facts = np.mean([ep['facts_shared'] for ep in episodes_data])
        avg_exhibits = np.mean([ep['exhibits_covered'] for ep in episodes_data])
        avg_engagement = np.mean([np.mean(ep['engagement_scores']) for ep in episodes_data])
        
        # Option diversity
        all_options = {}
        all_subactions = {}
        for ep in episodes_data:
            for opt, count in ep['option_usage'].items():
                all_options[opt] = all_options.get(opt, 0) + count
            for sub, count in ep['subaction_usage'].items():
                all_subactions[sub] = all_subactions.get(sub, 0) + count
        
        # Calculate entropy (diversity measure)
        option_probs = np.array(list(all_options.values())) / sum(all_options.values())
        option_entropy = -np.sum(option_probs * np.log(option_probs + 1e-10))
        
        print("\n[PERFORMANCE METRICS]")
        print("-" * 80)
        print(f"  Episodes Tested: {len(episodes_data)}")
        print(f"  Total Turns: {total_turns}")
        print(f"  Avg Reward per Episode: {avg_reward:.3f}")
        print(f"  Avg Reward per Turn: {avg_reward_per_turn:.3f}")
        print(f"  Reward Std Dev: {np.std([ep['total_reward'] for ep in episodes_data]):.3f}")
        
        print("\n[CONTENT DELIVERY]")
        print("-" * 80)
        print(f"  Avg Facts Shared: {avg_facts:.2f}")
        print(f"  Avg Exhibits Covered: {avg_exhibits:.2f}")
        print(f"  Facts per Episode Range: {min([ep['facts_shared'] for ep in episodes_data])} - {max([ep['facts_shared'] for ep in episodes_data])}")
        
        print("\n[ENGAGEMENT]")
        print("-" * 80)
        print(f"  Avg Engagement Score: {avg_engagement:.3f}")
        print(f"  Engagement Std Dev: {np.std([np.mean(ep['engagement_scores']) for ep in episodes_data]):.3f}")
        
        print("\n[POLICY ANALYSIS]")
        print("-" * 80)
        print(f"  Option Diversity (Entropy): {option_entropy:.3f} / {np.log(len(all_options)):.3f} (max)")
        print(f"  Option Usage Distribution:")
        for opt, count in sorted(all_options.items(), key=lambda x: -x[1]):
            pct = 100 * count / sum(all_options.values())
            print(f"    {opt:20s}: {count:3d} times ({pct:5.1f}%)")
        
        print(f"\n  Top Subactions:")
        for sub, count in sorted(all_subactions.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / sum(all_subactions.values())
            print(f"    {sub:25s}: {count:3d} times ({pct:5.1f}%)")
        
        # Quality grades
        print("\n" + "=" * 80)
        print("QUALITY GRADES")
        print("=" * 80)
        
        def grade(value, thresholds):
            """Grade based on thresholds (A, B, C, D, F)"""
            if value >= thresholds[0]: return "A"
            elif value >= thresholds[1]: return "B"
            elif value >= thresholds[2]: return "C"
            elif value >= thresholds[3]: return "D"
            else: return "F"
        
        reward_grade = grade(avg_reward_per_turn, [0.8, 0.6, 0.4, 0.2])
        facts_grade = grade(avg_facts, [5, 3, 2, 1])
        exhibits_grade = grade(avg_exhibits, [3, 2.5, 2, 1.5])
        engagement_grade = grade(avg_engagement, [0.8, 0.6, 0.4, 0.2])
        diversity_grade = grade(option_entropy / np.log(len(all_options)), [0.8, 0.6, 0.4, 0.2])
        
        print(f"  Reward Performance:     {reward_grade} ({avg_reward_per_turn:.3f})")
        print(f"  Content Coverage:       {facts_grade} ({avg_facts:.1f} facts)")
        print(f"  Exploration:            {exhibits_grade} ({avg_exhibits:.1f} exhibits)")
        print(f"  Visitor Engagement:     {engagement_grade} ({avg_engagement:.3f})")
        print(f"  Policy Diversity:       {diversity_grade} ({option_entropy:.2f})")
        
        # Overall grade
        grades_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        overall_score = np.mean([grades_map[g] for g in [reward_grade, facts_grade, exhibits_grade, engagement_grade, diversity_grade]])
        overall_grade = ['F', 'D', 'C', 'B', 'A'][int(overall_score)]
        
        print(f"\n  ** OVERALL GRADE: {overall_grade} ({overall_score:.2f}/4.0) **")
        print("=" * 80)
        
        return {
            'avg_reward': avg_reward,
            'avg_reward_per_turn': avg_reward_per_turn,
            'avg_facts': avg_facts,
            'avg_exhibits': avg_exhibits,
            'avg_engagement': avg_engagement,
            'option_entropy': option_entropy,
            'grades': {
                'reward': reward_grade,
                'facts': facts_grade,
                'exhibits': exhibits_grade,
                'engagement': engagement_grade,
                'diversity': diversity_grade,
                'overall': overall_grade
            }
        }
    
    def run_full_assessment(self, num_episodes=3, personas=None):
        """Run full quality assessment"""
        if personas is None:
            personas = ["Agreeable", "Disagreeable", "Curious"]
        
        print("\n" + "=" * 80)
        print("STARTING MODEL QUALITY ASSESSMENT")
        print("=" * 80)
        print(f"Testing {num_episodes} episodes with personas: {personas}")
        print("=" * 80)
        
        # Load model
        env, agent, simulator, checkpoint = self.load_model()
        
        # Run test episodes
        episodes_data = []
        for i in range(num_episodes):
            persona = personas[i % len(personas)]
            episode_data = self.run_test_episode(env, agent, simulator, persona=persona, verbose=True)
            episodes_data.append(episode_data)
            print()
        
        # Compute quality metrics
        metrics = self.compute_quality_metrics(episodes_data)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'checkpoint_info': {
                'timestamp': checkpoint.get('timestamp', 'N/A'),
                'episodes': checkpoint.get('total_episodes', 'N/A'),
                'avg_reward': checkpoint.get('avg_reward', 0)
            },
            'test_episodes': episodes_data,
            'metrics': metrics
        }
        
        output_path = f"models/quality_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Assessment results saved to: {output_path}")
        
        return results


def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("HRL MUSEUM AGENT - MODEL QUALITY ASSESSMENT")
    print("=" * 80)
    
    assessor = ModelQualityAssessment("models/trained_agent.pt")
    
    # Run 3 test episodes with different personas
    results = assessor.run_full_assessment(
        num_episodes=3,
        personas=["Agreeable", "Curious", "Disagreeable"]
    )
    
    print("\n" + "=" * 80)
    print("ASSESSMENT COMPLETE!")
    print("=" * 80)
    print(f"Overall Grade: {results['metrics']['grades']['overall']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

