#!/usr/bin/env python3
"""
Add Hallucination Detection to HRL Simulator
Detects when agent claims past conversations that didn't happen
"""
import sys, io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def add_hallucination_detection():
    with open("src/simulator/sim8_adapter.py", "r") as f:
        content = f.read()
    
    # Add hallucination detection method before _check_agent_relevance
    hallucination_method = '''    def _detect_hallucinations(self, agent_utterance: str) -> tuple:
        """Detect if agent claims conversations that didn't happen."""
        agent_lower = (agent_utterance or "").lower()
        
        # Detect memory-seeking language
        memory_patterns = ["recall", "remember", "discussed", "we talked about", "earlier", "before"]
        has_memory_claim = any(pattern in agent_lower for pattern in memory_patterns)
        
        if not has_memory_claim:
            return False, ""
        
        # If claiming past discussion but dialogue history is empty = hallucination!
        if not self.dialogue_history:
            return True, "first_turn_memory_claim"
        
        # Extract claimed topic from utterance
        import re
        topic_patterns = [
            r'(?:about|regarding|on)\s+([a-z\s]+?)(?:\?|\.)',
            r'our\s+(?:discussion|conversation)\s+([a-z\s]+?)(?:\?|\.)',
        ]
        
        claimed_topic = None
        for pattern in topic_patterns:
            match = re.search(pattern, agent_lower)
            if match:
                claimed_topic = match.group(1).strip()
                break
        
        if claimed_topic:
            # Search dialogue history for this topic
            history_text = " ".join([turn.get("text", "").lower() for turn in self.dialogue_history])
            if claimed_topic not in history_text:
                return True, f"topic_not_discussed_{claimed_topic}"
        
        return False, ""
'''
    
    # Insert before _check_agent_relevance
    if "_detect_hallucinations" not in content:
        insert_point = content.find("    def _check_agent_relevance(self")
        if insert_point > 0:
            content = content[:insert_point] + hallucination_method + "\n" + content[insert_point:]
            print("[OK] Added hallucination detection method")
    
    # Update _check_agent_relevance to use hallucination detection
    if "is_hallucinating, hallucination_reason = self._detect_hallucinations" not in content:
        # Find the start of relevance_score assignments
        old_check = """        # Check current AOI mention
        current_aoi_clean = self.current_aoi.lower().replace("_", " ") if self.current_aoi else ""
        current_exhibit_clean = self.current_exhibit.lower().replace("_", " ") if self.current_exhibit else ""
        
        aoi_mentioned = current_aoi_clean in agent_lower
        exhibit_mentioned = current_exhibit_clean in agent_lower
        
        if not (aoi_mentioned or exhibit_mentioned):
            relevance_score -= 0.3  # Penalty: not mentioning current focus"""
        
        new_check = """        # CHECK 0: Hallucination detection (NEW!)
        is_hallucinating, hallucination_reason = self._detect_hallucinations(agent_utterance)
        if is_hallucinating:
            relevance_score -= 0.5  # MAJOR PENALTY
            self.off_topic_strikes += 2  # Double strike
            self.engagement_level *= 0.4  # Significant drop
        
        # Check current AOI mention
        current_aoi_clean = self.current_aoi.lower().replace("_", " ") if self.current_aoi else ""
        current_exhibit_clean = self.current_exhibit.lower().replace("_", " ") if self.current_exhibit else ""
        
        aoi_mentioned = current_aoi_clean in agent_lower
        exhibit_mentioned = current_exhibit_clean in agent_lower
        
        if not (aoi_mentioned or exhibit_mentioned):
            relevance_score -= 0.3  # Penalty: not mentioning current focus"""
        
        content = content.replace(old_check, new_check)
        print("[OK] Integrated hallucination detection into relevance check")
    
    # Add logic to force confused response when hallucinating
    if "if is_hallucinating:" not in content:
        # Find generate_user_response and add hallucination response forcing
        old_response_type = """        # STEP 4: Choose response type based on context and agent quality
        rtype = self._determine_response_type_contextual(agent_utterance, answered_well)"""
        
        new_response_type = """        # STEP 4: Choose response type based on context and agent quality
        # Force confused response if agent hallucinated
        if is_hallucinating:
            rtype = "confusion"
        else:
            rtype = self._determine_response_type_contextual(agent_utterance, answered_well)"""
        
        content = content.replace(old_response_type, new_response_type)
        print("[OK] Force confused responses for hallucinations")
    
    with open("src/simulator/sim8_adapter.py", "w") as f:
        f.write(content)
    
    print("\n[SUCCESS] Hallucination detection added!")
    print("Now agent false claims will be detected and punished with:")
    print("  - 0.5 penalty to relevance score")
    print("  - 2x strike multiplier")
    print("  - 0.4x engagement level drop")
    print("  - Forced confused user response")

if __name__ == "__main__":
    add_hallucination_detection()
