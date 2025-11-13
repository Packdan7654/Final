"""
H2 Experiment: Learned Terminations for Explain

Tests hypothesis that learned terminations track engagement and intent,
with Explain persisting during high dwell and terminating early when
dwell falls or intent shifts.

Variant: Fixed-duration Explain vs learned terminations
"""

__all__ = ['env', 'train', 'evaluate']

