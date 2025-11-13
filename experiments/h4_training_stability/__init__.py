"""
H4 Experiment: Training Stability

Tests hypothesis that hierarchical actor-critic trains more smoothly
than flat actor-critic, with steadier learning curves and faster convergence.

Comparison: Hierarchical vs Flat actor-critic training dynamics
"""

__all__ = ['train', 'evaluate']

