"""
Experimental Framework for Hypothesis Testing

This package contains experimental variants for testing hypotheses H5 and H6:
- H5: State ablation (dialogue-act state vs full DialogueBERT state)
- H6: Transition reward shaping (with vs without transition probability)

Each experiment variant extends the base environment/training components
and provides hypothesis-specific evaluation metrics.
"""

__all__ = ['baseline', 'h5_state_ablation', 'h6_transition_reward', 'shared']

