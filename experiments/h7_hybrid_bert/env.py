"""
H7 Environment Variant: Hybrid BERT

Extends base MuseumDialogueEnv to use:
- Standard BERT for i_t (intent embedding - single utterance)
- DialogueBERT for c_t (dialogue context - multi-turn)

State: [f_t, h_t, i_t, c_t] where:
- f_t: focus vector (n_exhibits + 1)
- h_t: dialogue history (n_exhibits + 4)
- i_t: standard BERT intent embedding (64-d projected from 768-d)
- c_t: DialogueBERT context embedding (64-d projected from 768-d)
Total: 149-d (same as baseline)
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.env import MuseumDialogueEnv
from src.utils.dialoguebert_intent_recognizer import DialogueBERTIntentRecognizer
import numpy as np


class H7HybridBERTEnv(MuseumDialogueEnv):
    """
    Environment variant for H7 hypothesis: hybrid BERT approach.
    
    Uses standard BERT for single-utterance intent (i_t) and DialogueBERT
    with turn/role embeddings for multi-turn context (c_t).
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize H7 environment variant."""
        # Create two recognizer instances BEFORE super().__init__()
        # because super().__init__() calls reset() which calls _get_obs()
        # 1. Standard BERT for intent embedding (i_t)
        self.intent_recognizer = DialogueBERTIntentRecognizer(mode="standard_bert")
        
        # 2. DialogueBERT for dialogue context (c_t)
        self.context_recognizer = DialogueBERTIntentRecognizer(mode="dialoguebert")
        
        super().__init__(*args, **kwargs)
        
        print("[H7] Hybrid BERT: Standard BERT for i_t, DialogueBERT for c_t")
        print(f"[H7] State dimension: {self.observation_space.shape[0]}-d (same as baseline)")
    
    def _get_obs(self):
        """
        Construct observation with hybrid BERT approach (H7 variant).
        
        State: [f_t, h_t, i_t, c_t]
        - f_t: focus vector (n_exhibits + 1)
        - h_t: dialogue history (n_exhibits + 4)
        - i_t: standard BERT intent embedding (64-d) - single utterance
        - c_t: DialogueBERT context embedding (64-d) - multi-turn
        """
        # 1. Focus vector f_t (same as baseline)
        focus_snapshot = np.zeros(self.n_exhibits + 1)
        if self.focus > 0:
            focus_snapshot[self.focus - 1] = 1.0
        else:
            focus_snapshot[-1] = 1.0  # No focus
        
        # 2. Dialogue history vector h_t (same as baseline)
        history = np.zeros(self.n_exhibits + len(self.options))
        coverage = self._get_museum_exhibit_coverage()
        
        for i, exhibit_name in enumerate(self.exhibit_keys):
            completion_ratio = coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
            history[i] = completion_ratio
        
        total_actions = sum(self.actions_used.values()) or 1
        for i, opt in enumerate(self.options):
            history[self.n_exhibits + i] = self.actions_used[opt] / total_actions
        
        # 3. Intent embedding i_t (64-d projected from 768-d)
        # H7: Use STANDARD BERT for single utterance (no turn/role embeddings)
        # Get turn number from last user utterance in dialogue history
        current_turn = 0
        if self.dialogue_history:
            for entry in reversed(self.dialogue_history):
                if len(entry) >= 3 and entry[0] == "user":
                    current_turn = entry[2]
                    break
            if current_turn == 0 and self.dialogue_turn_counter > 0:
                current_turn = self.dialogue_turn_counter
        
        # Standard BERT for intent (turn_number ignored in standard_bert mode)
        intent_embedding_768 = self.intent_recognizer.get_intent_embedding(
            self.last_user_utterance, role="user", turn_number=current_turn
        )
        
        # Apply projection: i_t = P * e_t
        intent_embedding_64 = np.dot(self.projection_matrix, intent_embedding_768).astype(np.float32)
        
        # 4. Dialogue context c_t (64-d projected from 768-d)
        # H7: Use DIALOGUEBERT for multi-turn context (with turn/role embeddings)
        dialogue_context_768 = self.context_recognizer.get_dialogue_context(
            self.dialogue_history, max_turns=3
        )
        
        # Apply projection: c_t = P * context_768
        dialogue_context_64 = np.dot(self.projection_matrix, dialogue_context_768).astype(np.float32)
        
        # Track embeddings for insights (keep full 768-d for visualization)
        prev_intent = getattr(self, '_last_intent_embedding', np.zeros(768, dtype=np.float32))
        prev_context = getattr(self, '_last_dialogue_context', np.zeros(768, dtype=np.float32))
        self._prev_intent_embedding = prev_intent.astype(np.float32)
        self._prev_dialogue_context = prev_context.astype(np.float32)
        self._last_intent_embedding = intent_embedding_768.astype(np.float32)
        self._last_dialogue_context = dialogue_context_768.astype(np.float32)
        
        # Concatenate into observation vector: [f_t, h_t, i_t, c_t]
        # Total: (n_exhibits + 1) + (n_exhibits + 4) + 64 + 64
        obs = np.concatenate([
            focus_snapshot,        # (n_exhibits + 1)-d
            history,               # (n_exhibits + 4)-d
            intent_embedding_64,   # 64-d (standard BERT)
            dialogue_context_64    # 64-d (DialogueBERT)
        ]).astype(np.float32)
        
        return obs

