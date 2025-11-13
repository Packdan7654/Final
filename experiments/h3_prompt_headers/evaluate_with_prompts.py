"""
H3 Evaluation: Compare Structured vs Minimal Prompts

Uses baseline trained model, runs evaluation episodes with:
1. Structured prompts (baseline)
2. Minimal prompts (H3 variant)

Compares outputs to test if structured headers improve KB grounding.
NO TRAINING REQUIRED - just evaluation with different prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from pathlib import Path
import torch

from src.environment.env import MuseumDialogueEnv
from src.simulator.sim8_adapter import Sim8Simulator
from src.agent.actor_critic_agent import ActorCriticAgent
from src.utils.knowledge_graph import SimpleKnowledgeGraph
from experiments.shared.evaluation_framework import HypothesisEvaluator
import src.utils.dialogue_planner as dp


# Minimal prompt function (H3 variant)
def build_minimal_prompt(option, subaction, ex_id, last_utt, facts_all, facts_used, 
                        selected_fact=None, dialogue_history=None, exhibit_names=None,
                        knowledge_graph=None, target_exhibit=None, coverage_dict=None):
    """
    Minimal prompt without structured headers (H3 variant).
    
    Removes verbose formatting, rules, and examples, but keeps essential information:
    - Current exhibit and visitor's message
    - Available facts (for Explain actions)
    - Facts already used (to avoid repetition)
    - Dialogue history (for coherence)
    - Target exhibit (for transitions)
    """
    import re
    
    # Base context
    prompt_parts = []
    prompt_parts.append(f"You are a museum guide at: {ex_id.replace('_', ' ')}")
    prompt_parts.append(f"Visitor said: \"{last_utt}\"")
    prompt_parts.append(f"Action: {option} / {subaction}")
    prompt_parts.append("")
    
    # Add recent dialogue history (last 2 exchanges for coherence)
    if dialogue_history and len(dialogue_history) > 0:
        recent = dialogue_history[-4:] if len(dialogue_history) > 4 else dialogue_history
        prompt_parts.append("Recent conversation:")
        for role, utterance in recent:
            role_label = "Agent" if role == "agent" else "Visitor"
            prompt_parts.append(f"  {role_label}: \"{utterance}\"")
        prompt_parts.append("")
    
    # Extract fact IDs already mentioned (to avoid repetition)
    fact_ids_mentioned = set()
    if dialogue_history:
        for _, utterance in dialogue_history:
            fact_ids_mentioned.update(re.findall(r'\[([A-Z]{2}_\d{3})\]', utterance))
    
    # Route by option/subaction
    if option == "Explain":
        if subaction == "ExplainNewFact":
            # Show available facts (unmentioned)
            if facts_all:
                prompt_parts.append("Available facts to share:")
                for fact in facts_all[:5]:  # Limit to 5 to keep it minimal
                    prompt_parts.append(f"  {fact}")
                prompt_parts.append("")
                if fact_ids_mentioned:
                    prompt_parts.append(f"Already mentioned fact IDs (don't repeat): {sorted(fact_ids_mentioned)}")
                    prompt_parts.append("")
                prompt_parts.append("Share 1-3 facts that address what the visitor said. Include fact IDs in brackets like [TU_001].")
            else:
                prompt_parts.append("No new facts available. Ask if they'd like to explore a different aspect.")
        
        elif subaction == "RepeatFact":
            if facts_used:
                fact_to_repeat = selected_fact if selected_fact else facts_used[-1]
                fact_id_match = re.search(r'\[([A-Z]{2}_\d{3})\]', fact_to_repeat)
                fact_id = fact_id_match.group(1) if fact_id_match else ""
                prompt_parts.append(f"Rephrase this fact in new words: {fact_to_repeat}")
                prompt_parts.append(f"Must include fact ID: [{fact_id}]")
            else:
                prompt_parts.append("No facts shared yet. Share an interesting fact about this exhibit.")
        
        elif subaction == "ClarifyFact":
            if facts_used:
                fact_to_clarify = selected_fact if selected_fact else facts_used[-1]
                prompt_parts.append(f"Clarify this fact more simply: {fact_to_clarify}")
            else:
                prompt_parts.append("Clarify an interesting fact about this exhibit.")
    
    elif option == "AskQuestion":
        if subaction == "AskOpinion":
            prompt_parts.append("Ask for their opinion or feeling about what we discussed.")
        elif subaction == "AskMemory":
            prompt_parts.append("Ask if they remember something specific we discussed earlier.")
        elif subaction == "AskClarification":
            prompt_parts.append("Ask what specific aspect interests them most.")
    
    elif option == "OfferTransition":
        if target_exhibit:
            target_name = target_exhibit.replace('_', ' ')
            current_name = ex_id.replace('_', ' ') if ex_id else 'current exhibit'
            prompt_parts.append(f"Current: {current_name}")
            prompt_parts.append(f"Suggest moving to: {target_name}")
            if coverage_dict:
                current_stats = coverage_dict.get(ex_id, {"mentioned": 0, "total": 1})
                target_stats = coverage_dict.get(target_exhibit, {"mentioned": 0, "total": 1})
                prompt_parts.append(f"Current exhibit: {current_stats['mentioned']}/{current_stats['total']} facts covered")
                prompt_parts.append(f"Target exhibit: {target_stats['mentioned']}/{target_stats['total']} facts covered")
            prompt_parts.append("")
            prompt_parts.append(f"Suggest moving to {target_name} in a natural way (2 sentences).")
        else:
            prompt_parts.append("Suggest moving to a different exhibit.")
    
    elif option == "Conclude":
        if subaction == "WrapUp":
            prompt_parts.append("Thank them warmly for visiting and express hope they enjoyed it.")
        elif subaction == "SummarizeKeyPoints":
            if facts_used:
                key_points = facts_used[-3:] if len(facts_used) >= 3 else facts_used
                prompt_parts.append("Summarize these key points briefly:")
                for fact in key_points:
                    prompt_parts.append(f"  {fact}")
                prompt_parts.append("")
                prompt_parts.append("Recap 2-3 main points (2-3 sentences, no fact IDs).")
            else:
                prompt_parts.append("Provide a warm conclusion to the visit.")
    
    prompt_parts.append("")
    prompt_parts.append("Response (2-3 sentences):")
    
    return "\n".join(prompt_parts)


def run_evaluation_episodes(env, agent, simulator, num_episodes=500, use_minimal_prompts=False):
    """
    Run evaluation episodes with a trained model.
    
    Args:
        env: MuseumDialogueEnv
        agent: Trained ActorCriticAgent
        simulator: Sim8Simulator
        num_episodes: Number of evaluation episodes
        use_minimal_prompts: If True, use minimal prompts (H3), else structured (baseline)
    """
    # Patch prompt function if needed
    original_build_prompt = dp.build_prompt
    if use_minimal_prompts:
        dp.build_prompt = build_minimal_prompt
    
    all_turns = []
    all_episodes = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        simulator.initialize_session(persona="Agreeable")
        agent.reset()
        
        episode_turns = []
        episode_reward = 0.0
        turn_count = 0
        
        while turn_count < env.max_turns:
            # Agent selects action
            available_options = env._get_available_options()
            available_subactions_dict = {
                opt: env._get_available_subactions(opt) 
                for opt in available_options
            }
            
            action = agent.select_action(
                obs,
                available_options,
                available_subactions_dict,
                deterministic=True  # Use deterministic policy for evaluation
            )
            
            # Environment step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Get user response
            user_response = simulator.generate_user_response(
                agent_utterance=info.get("agent_utterance", ""),
                current_exhibit=info.get("current_exhibit", ""),
                option=info.get("option", ""),
                subaction=info.get("subaction", "")
            )
            
            # Update environment with simulator state
            env.dwell = user_response.get("gaze_features", [0.0])[0]
            
            # Store turn data
            turn_data = {
                'episode': episode,
                'turn': turn_count,
                'option': info.get('option'),
                'subaction': info.get('subaction'),
                'agent_utterance': info.get('agent_utterance', ''),
                'user_utterance': user_response.get('response_text', ''),
                'dwell': env.dwell,
                'reward': reward,
                'facts_mentioned_in_utterance': info.get('facts_mentioned_in_utterance', []),
                'hallucinated_facts': info.get('hallucinated_facts', []),
                'response_type': user_response.get('response_type', 'statement'),
            }
            episode_turns.append(turn_data)
            all_turns.append(turn_data)
            
            episode_reward += reward
            turn_count += 1
            
            if done or truncated:
                break
            
            obs = next_obs
        
        all_episodes.append({
            'episode': episode,
            'total_reward': episode_reward,
            'num_turns': turn_count,
            'turns': episode_turns
        })
    
    # Restore original prompt function
    dp.build_prompt = original_build_prompt
    
    return all_turns, all_episodes


def load_trained_model(baseline_dir, device='cpu'):
    """Load trained agent from baseline experiment."""
    baseline_path = Path(baseline_dir)
    
    # Find checkpoint
    checkpoint_files = list((baseline_path / 'checkpoints').glob('*.pt'))
    if not checkpoint_files:
        # Try models directory
        checkpoint_files = list((baseline_path / 'models').glob('*.pt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {baseline_path}")
    
    # Load latest checkpoint
    checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    # Load knowledge graph
    kg_path = baseline_path.parent.parent.parent / "museum_knowledge_graph.json"
    if not kg_path.exists():
        kg_path = Path("museum_knowledge_graph.json")
    
    kg = SimpleKnowledgeGraph(str(kg_path))
    env = MuseumDialogueEnv(knowledge_graph_path=str(kg_path), max_turns=40)
    
    # Initialize agent with same architecture
    state_dim = env.observation_space.shape[0]
    agent = ActorCriticAgent(
        state_dim=state_dim,
        options=env.options,
        subactions=env.subactions,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        device=device
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.network.load_state_dict(checkpoint['network_state_dict'])
    agent.eval()
    
    return env, agent, kg


def main():
    parser = argparse.ArgumentParser(description='Evaluate H3: Compare Prompts (No Training)')
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Path to baseline trained model directory')
    parser.add_argument('--num-episodes', type=int, default=500,
                       help='Number of evaluation episodes per condition')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        print(f"❌ Baseline directory not found: {baseline_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else Path('experiments/results/h3')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("H3 EVALUATION: STRUCTURED vs MINIMAL PROMPTS")
    print("=" * 80)
    print(f"Baseline model: {baseline_dir}")
    print(f"Episodes per condition: {args.num_episodes}")
    print("=" * 80)
    print()
    
    # Load trained model
    print("Loading trained baseline model...")
    env, agent, kg = load_trained_model(baseline_dir, device=args.device)
    simulator = Sim8Simulator(knowledge_graph=kg)
    
    # Run evaluation with STRUCTURED prompts (baseline)
    print("\n" + "=" * 80)
    print("Running evaluation with STRUCTURED prompts (baseline)...")
    print("=" * 80)
    structured_turns, structured_episodes = run_evaluation_episodes(
        env, agent, simulator, 
        num_episodes=args.num_episodes,
        use_minimal_prompts=False
    )
    
    # Run evaluation with MINIMAL prompts (H3)
    print("\n" + "=" * 80)
    print("Running evaluation with MINIMAL prompts (H3 variant)...")
    print("=" * 80)
    minimal_turns, minimal_episodes = run_evaluation_episodes(
        env, agent, simulator,
        num_episodes=args.num_episodes,
        use_minimal_prompts=True
    )
    
    # Compute metrics for both
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)
    
    # Create evaluators
    structured_eval = HypothesisEvaluator('baseline_structured', output_dir)
    structured_eval.turn_data = structured_turns
    structured_eval.metrics = {
        'episode_returns': [ep['total_reward'] for ep in structured_episodes],
        'episode_lengths': [ep['num_turns'] for ep in structured_episodes],
        'episode_coverage': []  # Will be computed
    }
    
    minimal_eval = HypothesisEvaluator('h3_minimal', output_dir)
    minimal_eval.turn_data = minimal_turns
    minimal_eval.metrics = {
        'episode_returns': [ep['total_reward'] for ep in minimal_episodes],
        'episode_lengths': [ep['num_turns'] for ep in minimal_episodes],
        'episode_coverage': []  # Will be computed
    }
    
    # Compute H3 metrics for both
    structured_h3_metrics = structured_eval.compute_h3_metrics()
    minimal_h3_metrics = minimal_eval.compute_h3_metrics()
    
    # Comparison
    comparison = {
        'grounding_precision': {
            'structured': structured_h3_metrics.get('grounding_precision', 0.0),
            'minimal': minimal_h3_metrics.get('grounding_precision', 0.0),
            'difference': structured_h3_metrics.get('grounding_precision', 0.0) - minimal_h3_metrics.get('grounding_precision', 0.0)
        },
        'grounding_recall': {
            'structured': structured_h3_metrics.get('grounding_recall', 0.0),
            'minimal': minimal_h3_metrics.get('grounding_recall', 0.0),
            'difference': structured_h3_metrics.get('grounding_recall', 0.0) - minimal_h3_metrics.get('grounding_recall', 0.0)
        },
        'hallucination_rate': {
            'structured': structured_h3_metrics.get('hallucination_rate', 0.0),
            'minimal': minimal_h3_metrics.get('hallucination_rate', 0.0),
            'difference': structured_h3_metrics.get('hallucination_rate', 0.0) - minimal_h3_metrics.get('hallucination_rate', 0.0)
        },
        'on_plan_compliance': {
            'structured': structured_h3_metrics.get('on_plan_compliance', {}),
            'minimal': minimal_h3_metrics.get('on_plan_compliance', {})
        },
        'act_classifier_agreement': {
            'structured': structured_h3_metrics.get('act_classifier_agreement', 0.0),
            'minimal': minimal_h3_metrics.get('act_classifier_agreement', 0.0),
            'difference': structured_h3_metrics.get('act_classifier_agreement', 0.0) - minimal_h3_metrics.get('act_classifier_agreement', 0.0)
        }
    }
    
    # Save results
    results = {
        'experiment_name': 'h3',
        'hypothesis': 'H3',
        'baseline_model': str(baseline_dir),
        'num_episodes': args.num_episodes,
        'structured_prompts': {
            'h3_metrics': structured_h3_metrics,
            'common_metrics': structured_eval.compute_common_metrics()
        },
        'minimal_prompts': {
            'h3_metrics': minimal_h3_metrics,
            'common_metrics': minimal_eval.compute_common_metrics()
        },
        'comparison': comparison
    }
    
    output_file = output_dir / 'h3_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("H3 EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print("\nKey Comparisons:")
    print(f"  Grounding Precision: Structured={comparison['grounding_precision']['structured']:.3f}, Minimal={comparison['grounding_precision']['minimal']:.3f}")
    print(f"  Hallucination Rate: Structured={comparison['hallucination_rate']['structured']:.3f}, Minimal={comparison['hallucination_rate']['minimal']:.3f}")
    print(f"  Act Agreement: Structured={comparison['act_classifier_agreement']['structured']:.3f}, Minimal={comparison['act_classifier_agreement']['minimal']:.3f}")


if __name__ == '__main__':
    main()

