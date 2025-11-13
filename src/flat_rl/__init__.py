"""
Flat RL variant for the museum dialogue agent.

This package mirrors the hierarchical setup but exposes a convenience API for
training and evaluating a flat policy whose action space is the set of all
individual subactions (Explain / Ask / Transition / Conclude primitives).
"""

from .env import FlatDialogueEnv  # noqa: F401
from .agent import FlatActorCriticAgent  # noqa: F401
from .trainer import FlatActorCriticTrainer  # noqa: F401
from .training_loop import FlatTrainingLoop  # noqa: F401


