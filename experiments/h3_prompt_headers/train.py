"""
Training script for H3: Minimal Prompts (no structured headers)

Trains agent with minimal prompts instead of structured headers.
Tests if structured headers improve KB grounding and faithfulness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import json
from datetime import datetime
from src.training.training_loop import HRLTrainingLoop


# Monkey-patch dialogue planner for H3 variant
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


class H3TrainingLoop(HRLTrainingLoop):
    """Training loop for H3 with minimal prompts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Patch dialogue planner to use minimal prompts
        import src.utils.dialogue_planner as dp
        dp.build_prompt = build_minimal_prompt
        
        self.training_title = "H3 EXPERIMENT: MINIMAL PROMPTS (NO HEADERS)"


def main():
    parser = argparse.ArgumentParser(description='Train H3: Minimal Prompts')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--name', type=str, default='h3_minimal_prompts')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    from train import create_experiment_folder
    exp_dir, exp_num = create_experiment_folder(args.name, 'major')
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    
    metadata = {
        "experiment_number": exp_num,
        "experiment_name": args.name,
        "hypothesis": "H3",
        "variant": "minimal_prompts",
        "timestamp": datetime.now().isoformat(),
        "episodes": args.episodes,
        "max_turns_per_episode": args.turns,
        "device": args.device,
        "learning_rate": args.lr,
        "gamma": args.gamma,
    }
    
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 80)
    print("H3 EXPERIMENT: MINIMAL PROMPTS")
    print("=" * 80)
    print(f"Testing: Minimal prompts vs structured headers")
    print(f"Episodes: {args.episodes}")
    print("=" * 80)
    print()
    
    training_loop = H3TrainingLoop(
        max_episodes=args.episodes,
        max_turns_per_episode=args.turns,
        knowledge_graph_path="museum_knowledge_graph.json",
        learning_rate=args.lr,
        gamma=args.gamma,
        use_actor_critic=True,
        device=args.device,
        verbose=args.verbose
    )
    
    training_loop.run_training()
    
    print("\n✅ H3 training complete!")
    print(f"Results: {exp_dir}")


if __name__ == '__main__':
    main()

