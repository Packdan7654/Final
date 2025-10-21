#!/usr/bin/env python3
"""
Apply HRL Simulator Enhancements
Adds dialogue history, disengagement, and context awareness
"""
import re
import sys, io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def apply_enhancements():
    with open("src/simulator/sim8_adapter.py", "r") as f:
        content = f.read()
    
    # Enhancement 1: Initialize dialogue history
    if "self.dialogue_history = []" not in content:
        content = content.replace(
            "self.last_user_response = {}\n\n    def get_current_aoi",
            """self.last_user_response = {}
        
        # Initialize dialogue history and learning
        self.dialogue_history = []
        self.facts_learned = set()
        self.exhibits_visited = set()
        self.max_history_length = 8

    def get_current_aoi"""
        )
        print("[OK] Added dialogue history initialization")
    
    # Enhancement 2: Add relevance check and history tracking
    if "# STEP 0: Check agent relevance" not in content:
        content = content.replace(
            "# Reset silence streak\n        self.consecutive_silence_count = 0\n\n        # STEP 1: Check if agent answered",
            """# Reset silence streak
        self.consecutive_silence_count = 0

        # STEP 0: Check agent relevance to current context
        relevance_score = self._check_agent_relevance(agent_utterance)
        
        # Track agent utterance in dialogue history
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history.pop(0)
        
        # STEP 1: Check if agent answered"""
        )
        print("[OK] Added relevance check and dialogue history tracking")
    
    # Enhancement 3: Apply engagement level to gaze synthesis
    if "engagement_bonus *= self.engagement_level" not in content:
        content = content.replace(
            """        # Base engagement modifiers
        engagement_bonus = 0.0
        if answered_well:
            engagement_bonus = 0.2  # Boost engagement when agent is helpful
        else:
            engagement_bonus = -0.25  # Reduce engagement when agent is off-topic""",
            """        # Base engagement modifiers
        engagement_bonus = 0.0
        if answered_well:
            engagement_bonus = 0.2  # Boost engagement when agent is helpful
        else:
            engagement_bonus = -0.25  # Reduce engagement when agent is off-topic
        
        # Apply engagement level multiplier from disengagement tracking
        engagement_bonus *= self.engagement_level"""
        )
        print("[OK] Applied engagement level multiplier to gaze synthesis")
    
    # Enhancement 4: Track visited exhibits and facts in response
    if "self.exhibits_visited.add(self.current_exhibit)" not in content:
        content = content.replace(
            "self.last_user_response = response\n        return response",
            """self.last_user_response = response
        
        # Track in memory (NEW!)
        self.exhibits_visited.add(self.current_exhibit)
        self.dialogue_history.append({"role": "user", "text": utterance or ""})
        
        # Extract fact IDs if present
        fact_ids = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance)
        for fact_id in fact_ids:
            self.facts_learned.add(fact_id)
        
        return response"""
        )
        print("[OK] Added exhibit and fact tracking to responses")
    
    # Write updated file
    with open("src/simulator/sim8_adapter.py", "w") as f:
        f.write(content)
    
    print("\n[SUCCESS] All enhancements applied!")
    print("\nNext: python train.py --episodes 2 --turns 5")

if __name__ == "__main__":
    apply_enhancements()
