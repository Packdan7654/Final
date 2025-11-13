"""
H6 Experiment: Transition Reward Shaping

Tests hypothesis that transition probability-based reward shaping
improves transition timing and overall performance.

Variant: Removes transition insufficiency/sufficiency rewards
to test if probabilistic transition acceptance alone is sufficient.
"""

__all__ = ['env', 'train', 'evaluate']

