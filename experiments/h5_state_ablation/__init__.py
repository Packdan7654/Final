"""
H5 Experiment: State Ablation

Tests hypothesis that a compact dialogue-act state representation
can match or exceed performance of full DialogueBERT embeddings.

State reduction: 149-d (full DialogueBERT) -> ~20-d (dialogue acts + focus + history)
"""

__all__ = ['env', 'train', 'evaluate']

