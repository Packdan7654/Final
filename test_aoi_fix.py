#!/usr/bin/env python3
"""
Test script for AOI detection fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulator.sim8_adapter import Sim8Simulator

def test_aoi_detection():
    """Test AOI detection improvements"""
    print("Testing AOI Detection Improvements...")

    sim = Sim8Simulator()
    print(f'Updated simulator exhibits: {sorted(sim.exhibits)}')

    test_cases = [
        'Let\'s visit the Gemstones exhibit',
        'I want to see the Necklace next',
        'Let\'s check out the Cavalier hat',
        'The Ring looks interesting',
        'gemstones',
        'necklace',
        'turban',
        'king caspar'
    ]

    for utterance in test_cases:
        detected_aoi, detected_parent = sim._detect_aoi_and_parent(utterance)
        detected_exhibit = sim.PARENT_TO_EXHIBIT.get(detected_parent, 'Unknown') if detected_parent else 'None'
        if detected_aoi in sim.AOI_TO_EXHIBIT:
            detected_exhibit = sim.AOI_TO_EXHIBIT[detected_aoi]
        print(f'  "{utterance}" -> AOI: {detected_aoi}, Parent: {detected_parent}, Exhibit: {detected_exhibit}')

if __name__ == "__main__":
    test_aoi_detection()
