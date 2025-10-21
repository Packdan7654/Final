#!/usr/bin/env python3
"""
Test script for improved transition logic in HRL Museum Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulator.sim8_adapter import Sim8Simulator
from src.environment.env import MuseumDialogueEnv

def test_aoi_detection():
    """Test AOI detection improvements"""
    print("Testing AOI Detection Improvements...")

    simulator = Sim8Simulator()

    # Test cases for better exhibit name detection
    test_cases = [
        ("Let's visit the King Caspar exhibit", "King_Caspar"),
        ("Let's check out the Cavalier hat", "Cavalier_hat"),
        ("I want to see the Ivory tusk next", "Ivory_tusk"),
        ("Let's go to the Box exhibit", "Box"),
        ("caspar", "King_Caspar"),  # Partial match
        ("hat", "Cavalier_hat"),    # Partial match
        ("tusk", "Ivory_tusk"),     # Partial match
    ]

    for utterance, expected_exhibit in test_cases:
        detected_aoi, detected_parent = simulator._detect_aoi_and_parent(utterance)
        detected_exhibit = simulator.PARENT_TO_EXHIBIT.get(detected_parent, None)

        print(f"  '{utterance}' -> {detected_exhibit} (expected: {expected_exhibit})")
        if detected_exhibit == expected_exhibit:
            print("    PASS")
        else:
            print(f"    FAIL (got {detected_exhibit}, expected {expected_exhibit})")

    print()

def test_transition_determinism():
    """Test that transitions are deterministic when suggesting movement"""
    print("Testing Transition Determinism...")

    simulator = Sim8Simulator()
    simulator.initialize_session("Agreeable")

    # Test that transition suggestions are followed deterministically
    transition_utterances = [
        "Let's visit the Turban exhibit next",
        "Shall we check out the Cavalier hat?",
        "I think you'd love the Ivory tusk - let's go there",
    ]

    for utterance in transition_utterances:
        initial_exhibit = simulator.current_exhibit
        initial_aoi = simulator.current_aoi

        # Generate response (this should trigger transition detection)
        response = simulator.generate_user_response(utterance)

        final_exhibit = simulator.current_exhibit
        final_aoi = simulator.current_aoi

        print(f"  '{utterance}'")
        print(f"    Before: {initial_exhibit} ({initial_aoi})")
        print(f"    After:  {final_exhibit} ({final_aoi})")

        # Check if transition occurred (should be deterministic for movement suggestions)
        if "visit" in utterance.lower() or "check out" in utterance.lower() or "go there" in utterance.lower():
            if final_exhibit != initial_exhibit:
                print("    PASS - Transition occurred")
            else:
                print("    FAIL - No transition occurred")
        else:
            print("    SKIP - Not a clear transition suggestion")

    print()

def test_state_influence():
    """Test that state information influences simulator AOI changes"""
    print("Testing State Influence on Simulator...")

    simulator = Sim8Simulator()
    simulator.initialize_session("Agreeable")

    print(f"  Initial state: {simulator.current_exhibit} ({simulator.current_aoi})")

    # Test state influence
    simulator.update_from_state(2)  # Focus on exhibit at index 1 (Turban)

    print(f"  After state update: {simulator.current_exhibit} ({simulator.current_aoi})")

    if simulator.current_exhibit == "King_Caspar":
        print("    PASS - State influence worked")
    else:
        print("    FAIL - State influence didn't work")

    print()

def test_offer_transition_actions():
    """Test that new OfferTransition subactions work"""
    print("Testing OfferTransition Subactions...")

    env = MuseumDialogueEnv()

    # Test each new subaction
    subactions = ["SuggestMove", "LinkToOtherExhibit", "CheckReadiness"]

    for subaction in subactions:
        try:
            # Get target exhibit for transition
            current_exhibit = env._get_current_exhibit()
            target_exhibit = env._select_least_discussed_exhibit(current_exhibit)

            # Build prompt for this subaction
            try:
                prompt = env._generate_agent_response("OfferTransition", subaction)

                print(f"  {subaction}:")
                print(f"    Target exhibit: {target_exhibit}")
                print(f"    Prompt length: {len(prompt)} characters")
                print(f"    Contains target: {target_exhibit.replace('_', ' ').lower() in prompt.lower()}")
                print("    PASS")
            except Exception as e:
                print(f"  {subaction}: FAIL - {e}")

        except Exception as e:
            print(f"  {subaction}: FAIL - {e}")

    print()

if __name__ == "__main__":
    print("Testing HRL Museum Agent Transition Improvements")
    print("=" * 60)

    test_aoi_detection()
    test_transition_determinism()
    test_state_influence()
    test_offer_transition_actions()

    print("All tests completed!")
