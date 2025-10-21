"""
RQ1 Experiment: Long-Horizon Coherence

Research Question (from paper.tex):
"Can an option-based manager maintain long-horizon coherence and keep turns 
aligned with the intended strategy?"

Tests:
- Do Explain/Ask persist (≥3 turns)?
- Are Transition/Conclude brief (1-2 turns)?
- Do subactions align with options?
- Is dialogue coherent without drift?

Usage:
    python experiments/rq1_experiment.py
    python experiments/rq1_experiment.py --model models/my_model.pt --episodes 20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import json
from datetime import datetime
from src.training.training_loop import HRLTrainingLoop
from src.metrics.metrics_collector import MetricsCollector
from src.environment.env import MuseumDialogueEnv


def run_rq1_experiment(model_path, num_episodes=20, device='cpu'):
    """Run RQ1 experiment on trained model."""
    
    print("=" * 80)
    print("RQ1 EXPERIMENT: Long-Horizon Coherence")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 80)
    print()
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train first: python train.py --episodes 200")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"✓ Loaded trained model from {checkpoint.get('timestamp', 'Unknown')}")
    print()
    
    # Initialize environment and metrics
    env = MuseumDialogueEnv(knowledge_graph_path="museum_knowledge_graph.json")
    metrics_collector = MetricsCollector(
        output_dir="results/rq1",
        knowledge_graph=env.knowledge_graph
    )
    
    # Initialize evaluation loop
    eval_loop = HRLTrainingLoop(
        max_episodes=num_episodes,
        max_turns_per_episode=30,
        knowledge_graph_path="museum_knowledge_graph.json",
        use_actor_critic=True,
        device=device,
        turn_delay=0.0
    )
    
    # Load weights
    eval_loop.agent.network.load_state_dict(checkpoint['agent_state_dict'])
    eval_loop.agent.network.eval()
    
    # Run episodes with metrics collection
    print("Running evaluation episodes with RQ1 metrics...")
    print()
    
    for episode in range(num_episodes):
        metrics_collector.start_episode()
        
        # Run episode
        obs, info = eval_loop.env.reset()
        eval_loop.simulator.initialize_session(persona="Agreeable")
        eval_loop.agent.reset()
        
        for turn in range(30):
            # Update environment state
            current_exhibit = eval_loop.simulator.get_current_aoi()
            focus = 0
            if current_exhibit in eval_loop.env.exhibit_keys:
                focus = eval_loop.env.exhibit_keys.index(current_exhibit) + 1
            eval_loop.env.update_user_state(focus=focus, utterance="")
            
            # Get action
            available_options = eval_loop.env._get_available_options()
            if not available_options:
                break
            
            available_subactions_dict = {}
            for opt in available_options:
                available_subactions_dict[opt] = eval_loop.env._get_available_subactions(opt)
            
            action_info = eval_loop.agent.select_action(
                state=obs,
                available_options=available_options,
                available_subactions_dict=available_subactions_dict,
                deterministic=True  # Deterministic for evaluation
            )
            
            action = {
                "option": action_info['option'],
                "subaction": action_info['subaction'],
                "terminate_option": action_info['terminated']
            }
            
            # Step
            next_obs, reward, done, truncated, info = eval_loop.env.step(action)
            
            # Extract facts
            agent_utterance = info.get("agent_utterance", "")
            current_exhibit = info.get("current_exhibit", "")
            if agent_utterance and current_exhibit:
                mentioned_facts = eval_loop.env.extract_facts_from_agent_utterance(agent_utterance, current_exhibit)
                info["facts_mentioned_in_utterance"] = mentioned_facts
            
            # Add DialogueBERT insights
            eval_loop.env.add_dialoguebert_insights_to_info(info)
            
            # Update simulator
            if agent_utterance:
                user_response = eval_loop.simulator.generate_user_response(agent_utterance)
                if user_response.get("utterance"):
                    eval_loop.env.update_user_state(utterance=user_response["utterance"])
                gaze_feats = user_response.get("gaze_features") or []
                if gaze_feats:
                    eval_loop.env.update_user_state(dwell=float(gaze_feats[0]))
            
            # Collect metrics
            step_data = {
                'turn': turn + 1,
                'option': info.get('option', 'Unknown'),
                'subaction': info.get('subaction', 'Unknown'),
                'terminated': info.get('terminated_option', False),
                'agent_utterance': agent_utterance,
                'dwell': info.get('dwell', 0.5),
                'focus': info.get('current_focus', 0),
                'current_exhibit': current_exhibit,
                'intent_category': info.get('dialoguebert_insights', {}).get('intent_category', 'unknown'),
                'facts_mentioned': info.get('facts_mentioned_in_utterance', []),
                'reward': reward
            }
            metrics_collector.add_step(step_data)
            
            obs = next_obs
            if done:
                break
        
        # End episode
        episode_metrics = metrics_collector.end_episode(episode_num=episode+1)
        
        # Print progress
        if episode_metrics and 'rq1' in episode_metrics:
            rq1 = episode_metrics['rq1']
            persistence = rq1.get('option_persistence', {})
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Explain={persistence.get('explain_persistence', 0):.1f}, "
                  f"Ask={persistence.get('ask_persistence', 0):.1f}, "
                  f"Transition={persistence.get('transition_duration', 0):.1f}, "
                  f"Conclude={persistence.get('conclude_duration', 0):.1f}")
    
    # Generate final report
    print()
    print("=" * 80)
    print("COMPUTING FINAL RQ1 METRICS...")
    print("=" * 80)
    
    final_report = metrics_collector.generate_final_report()
    
    # Save results
    os.makedirs("results/rq1", exist_ok=True)
    results_path = "results/rq1/rq1_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Print summary
    print()
    if 'rq1' in final_report:
        rq1 = final_report['rq1']
        overall = rq1.get('overall_assessment', {})
        
        print("RQ1 RESULTS:")
        print("-" * 80)
        print(f"Composite Score: {overall.get('composite_score', 0):.3f}")
        print(f"Verdict: {overall.get('verdict', 'N/A')}")
        print(f"Recommendation: {overall.get('recommendation', 'N/A')}")
        print()
        
        persistence = rq1.get('option_persistence', {})
        print("Option Persistence:")
        print(f"  Explain:    {persistence.get('explain_persistence', 0):.1f} turns {_check_threshold(persistence.get('explain_persistence', 0), 3.0, True)}")
        print(f"  Ask:        {persistence.get('ask_persistence', 0):.1f} turns {_check_threshold(persistence.get('ask_persistence', 0), 2.5, True)}")
        print(f"  Transition: {persistence.get('transition_duration', 0):.1f} turns {_check_threshold(persistence.get('transition_duration', 0), 2.0, False)}")
        print(f"  Conclude:   {persistence.get('conclude_duration', 0):.1f} turns {_check_threshold(persistence.get('conclude_duration', 0), 2.0, False)}")
        print()
        
        alignment = rq1.get('strategy_alignment', {})
        print(f"Strategy Alignment: {alignment.get('alignment_score', 0):.3f}")
        print()
        
        coherence = rq1.get('coherence', {})
        print(f"Coherence Score: {coherence.get('appropriate_transition_rate', 0):.3f}")
        print()
        
        diversity = rq1.get('diversity', {})
        print(f"Diversity (Entropy): {diversity.get('normalized_entropy', 0):.3f}")
        print()
        
        print("Strengths:")
        for strength in overall.get('key_strengths', [])[:3]:
            print(f"  ✓ {strength}")
        print()
        
        print("Weaknesses:")
        for weakness in overall.get('key_weaknesses', [])[:3]:
            print(f"  ✗ {weakness}")
    
    print()
    print("=" * 80)
    print(f"✓ Results saved to: {results_path}")
    print("=" * 80)
    
    return final_report


def _check_threshold(value, threshold, should_be_higher):
    """Helper to check if value meets threshold."""
    if should_be_higher:
        return "✓" if value >= threshold else "✗"
    else:
        return "✓" if value <= threshold else "✗"


def main():
    parser = argparse.ArgumentParser(description='RQ1 Experiment: Long-Horizon Coherence')
    parser.add_argument('--model', type=str, default='models/trained_agent.pt',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    run_rq1_experiment(
        model_path=args.model,
        num_episodes=args.episodes,
        device=args.device
    )


if __name__ == '__main__':
    main()

