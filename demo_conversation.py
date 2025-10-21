"""
Clean Demo Conversation with Trained Agent

Shows 5 turns of natural museum conversation
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from interact import MuseumAgentInterface

def print_separator(char='='):
    print(char * 80)

def main():
    # Initialize
    interface = MuseumAgentInterface()
    
    print_separator()
    print("DEMONSTRATION: 5-TURN MUSEUM CONVERSATION")
    print_separator()
    
    conversations = [
        {
            'turn': 1,
            'desc': 'Visitor arrives, looks at King Caspar painting',
            'utterance': 'Hello! This painting caught my eye. What can you tell me about it?',
            'focus': 1,  # King_Caspar
            'dwell': 0.82
        },
        {
            'turn': 2,
            'desc': 'Visitor shows high engagement, wants more details',
            'utterance': 'That\'s really interesting! Tell me more about the artist.',
            'focus': 1,  # Still King_Caspar
            'dwell': 0.95
        },
        {
            'turn': 3,
            'desc': 'Visitor curious about other pieces',
            'utterance': 'What other pieces from this period do you have?',
            'focus': 1,  # Still looking at King_Caspar
            'dwell': 0.68
        },
        {
            'turn': 4,
            'desc': 'Visitor transitions to Necklace',
            'utterance': 'This necklace looks amazing! Tell me about it.',
            'focus': 4,  # Necklace
            'dwell': 0.91
        },
        {
            'turn': 5,
            'desc': 'Visitor wants more details on necklace',
            'utterance': 'What materials is it made from?',
            'focus': 4,  # Necklace
            'dwell': 0.89
        }
    ]
    
    for conv in conversations:
        print(f"\n{'='*80}")
        print(f"TURN {conv['turn']}: {conv['desc']}")
        print(f"{'='*80}")
        
        print(f"\n[Visitor Input]")
        print(f"  Says: \"{conv['utterance']}\"")
        exhibit = interface.env.exhibit_keys[conv['focus']-1] if conv['focus'] > 0 else 'None'
        print(f"  Looking at: [{conv['focus']}] {exhibit}")
        print(f"  Engagement (dwell): {conv['dwell']:.2f}")
        
        try:
            response, info = interface.process_turn(
                utterance=conv['utterance'],
                focus=conv['focus'],
                dwell=conv['dwell']
            )
            
            print(f"\n[Agent Response]")
            print(f"  Decision: {info['option']} -> {info['subaction']}")
            print(f"  Says: \"{response}\"")
            
            print(f"\n[Turn Results]")
            print(f"  Reward: {info['total_reward']:.3f}")
            print(f"    - Engagement: {info['reward_engagement']:.3f}")
            print(f"    - Novelty: {info['reward_novelty']:.3f}")
            print(f"  Facts shared this turn: {info['facts_shared']}")
            if info.get('facts_mentioned_in_utterance'):
                print(f"  Fact IDs: {info['facts_mentioned_in_utterance']}")
            print(f"  Session total reward: {interface.env.session_reward:.3f}")
            
        except Exception as e:
            print(f"\n[ERROR]: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print("CONVERSATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total turns: {interface.env.turn_count}")
    print(f"Total reward: {interface.env.session_reward:.3f}")
    print(f"Avg reward per turn: {interface.env.session_reward / interface.env.turn_count:.3f}")
    
    total_facts = sum(len(facts) for facts in interface.env.facts_mentioned_per_exhibit.values())
    print(f"Total facts shared: {total_facts}")
    
    exhibits_visited = sum(1 for exp in interface.env.explained if exp > 0)
    print(f"Exhibits visited: {exhibits_visited}")
    
    print(f"\nAction distribution:")
    for action, count in interface.env.actions_used.items():
        if count > 0:
            pct = 100 * count / interface.env.turn_count
            print(f"  {action}: {count} times ({pct:.1f}%)")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

