"""
Dialogue Act Classifier for State Ablation (H5)

Uses HuggingFace zero-shot classification to map utterances to dialogue act types
with probability distributions, replacing full DialogueBERT embeddings for hypothesis testing.

Dialogue act types (8 categories):
- question: Visitor asks a question
- statement: Visitor makes a statement/observation
- acknowledgment: Visitor acknowledges/agrees
- confusion: Visitor expresses confusion
- follow_up: Visitor asks follow-up question
- directive: Visitor requests action (e.g., "show me", "let's go")
- clarification: Visitor asks for clarification (e.g., "what do you mean?")
- silence: No response
"""

import numpy as np
from typing import Optional
import warnings

# Suppress transformers warnings if needed
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not available. Install with: pip install transformers")


class DialogueActClassifier:
    """
    Pre-trained dialogue act classifier using HuggingFace zero-shot classification.
    
    Uses BART-large-MNLI model to classify utterances into dialogue act types
    with probability distributions (soft labels) instead of hard one-hot encoding.
    
    For H5 hypothesis testing, this provides a compact state representation
    (8-d probability distribution) instead of full DialogueBERT embeddings.
    """
    
    ACT_TYPES = [
        "question",
        "statement", 
        "acknowledgment",
        "confusion",
        "follow_up",
        "directive",
        "clarification",
        "silence"
    ]
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize dialogue act classifier with zero-shot classification pipeline.
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
                       Default: "facebook/bart-large-mnli" (high accuracy)
                       Alternative: "typeform/distilbert-base-uncased-mnli" (faster)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        # Initialize zero-shot classification pipeline
        # This uses Natural Language Inference (NLI) to score how well
        # the utterance matches each dialogue act label
        # Model will be downloaded on first use if not cached
        print(f"[H5] Loading zero-shot classifier: {model_name} (this may take a moment on first run)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1  # Use CPU (set to 0 for GPU if available)
        )
        
        print(f"[H5] Zero-shot classifier initialized: {model_name}")
        print(f"[H5] Dialogue act categories (8): {', '.join(self.ACT_TYPES)}")
        print(f"[H5] Using soft probability distributions (not one-hot)")
    
    def classify(self, utterance: Optional[str], 
                 previous_act: Optional[str] = None) -> str:
        """
        Classify utterance into dialogue act type (returns top label).
        
        Args:
            utterance: Visitor utterance text (None for silence)
            previous_act: Previous dialogue act (for context - not used in zero-shot,
                         but kept for interface compatibility)
            
        Returns:
            Dialogue act type string (top label)
        """
        if not utterance or not utterance.strip():
            return "silence"
        
        # Use zero-shot classification
        result = self.classifier(utterance, candidate_labels=self.ACT_TYPES)
        
        # Return top label (highest probability)
        return result['labels'][0]
    
    def classify_with_probabilities(self, utterance: Optional[str],
                                    previous_act: Optional[str] = None) -> dict:
        """
        Classify utterance and return full probability distribution.
        
        Args:
            utterance: Visitor utterance text (None for silence)
            previous_act: Previous dialogue act (for context - not used in zero-shot,
                         but kept for interface compatibility)
        
        Returns:
            Dictionary with 'label' (top label) and 'probabilities' (8-d array)
        """
        if not utterance or not utterance.strip():
            # Return silence with certainty
            probs = np.zeros(len(self.ACT_TYPES), dtype=np.float32)
            probs[self.ACT_TYPES.index("silence")] = 1.0
            return {
                'label': 'silence',
                'probabilities': probs
            }
        
        # Use zero-shot classification
        result = self.classifier(utterance, candidate_labels=self.ACT_TYPES)
        
        # Reorder probabilities to match ACT_TYPES order
        probabilities = np.zeros(len(self.ACT_TYPES), dtype=np.float32)
        for i, label in enumerate(self.ACT_TYPES):
            if label in result['labels']:
                idx = result['labels'].index(label)
                probabilities[i] = result['scores'][idx]
        
        return {
            'label': result['labels'][0],  # Top label
            'probabilities': probabilities  # 8-d probability distribution
        }
    
    def act_to_vector(self, act_type: str) -> np.ndarray:
        """
        Convert dialogue act type to one-hot vector (for backward compatibility).
        
        Note: This method is kept for compatibility, but H5 now uses
        classify_with_probabilities() to get soft probability distributions.
        
        Args:
            act_type: Dialogue act type string
            
        Returns:
            One-hot vector of shape (len(ACT_TYPES),)
        """
        if act_type not in self.ACT_TYPES:
            act_type = "statement"  # Default fallback
        
        vector = np.zeros(len(self.ACT_TYPES), dtype=np.float32)
        idx = self.ACT_TYPES.index(act_type)
        vector[idx] = 1.0
        
        return vector
    
    def get_state_dim(self) -> int:
        """Get dimension of dialogue act state vector."""
        return len(self.ACT_TYPES)  # 8-d


# Global instance
_classifier_instance = None

def get_dialogue_act_classifier() -> DialogueActClassifier:
    """
    Get global dialogue act classifier instance.
    
    Returns:
        DialogueActClassifier instance (singleton)
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DialogueActClassifier()
    return _classifier_instance
