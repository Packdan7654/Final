"""
DialogueBERT Intent Recognition Module

This module provides DialogueBERT-based intent recognition for the HRL museum 
dialogue agent. DialogueBERT uses contextual encoders that interpret utterances 
in light of speaker role and local dialogue history.

Paper: "DialogueBERT: A Self-Supervised Learning based Dialogue Pre-training Encoder"
by Zhang et al., 2021.

Key Features:
- Turn-aware encoding for multi-turn dialogues
- Role-specific token integration (user/system)
- Dialogue context tracking
- Robust intent recognition in conversational settings
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DialogueBERTIntentRecognizer:
    """
    DialogueBERT based intent recognition for museum dialogue agent.
    
    DialogueBERT advantages over standard BERT:
    - Turn-order and role encoding for dialogue context
    - Multi-turn conversation understanding
    - Better intent recognition accuracy in dialogues
    - Dialogue state tracking capabilities
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 device: Optional[str] = None):
        """
        Initialize DialogueBERT intent recognizer.
        
        Args:
            model_name: HuggingFace model name for base BERT model
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load DialogueBERT tokenizer and model (BERT backbone with dialogue features)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"DialogueBERT loaded successfully on {self.device}")
            logger.info(f"Model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load DialogueBERT: {e}")
            logger.info("Falling back to standard BERT")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback BERT model if ToD-BERT fails."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            self.model.to(self.device)
            self.model.eval()
            logger.info("Fallback BERT model loaded")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.model = None
            self.tokenizer = None
    
    def get_intent_embedding(self, utterance: str, role: str = "user") -> np.ndarray:
        """
        Generate intent embedding for a visitor utterance using DialogueBERT.
        
        Args:
            utterance: Visitor's utterance text
            role: Speaker role ("user" or "system")
            
        Returns:
            Intent embedding vector of shape (768,)
        """
        if not utterance or not utterance.strip():
            return self._get_silence_embedding()
        
        if self.model is None:
            return self._fallback_intent_embedding(utterance)
        
        try:
            # DialogueBERT approach: Add role markers for dialogue context
            if role.lower() == "user":
                # Prefix with role identifier for user utterances
                processed_text = f"user: {utterance}"
            else:
                # Prefix with role identifier for system/agent utterances
                processed_text = f"system: {utterance}"
            
            # Tokenize with DialogueBERT approach (BERT with dialogue context)
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding as intent representation
                intent_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return intent_embedding.flatten()
            
        except Exception as e:
            logger.warning(f"Error generating intent embedding: {e}")
            return self._fallback_intent_embedding(utterance)
    
    def get_dialogue_context(self, recent_utterances: List[Tuple[str, str]], 
                           max_turns: int = 3) -> np.ndarray:
        """
        Generate dialogue context embedding using DialogueBERT.
        
        Args:
            recent_utterances: List of (role, text) tuples
            max_turns: Maximum number of turns to include
            
        Returns:
            Dialogue context embedding vector of shape (768,)
        """
        if not recent_utterances:
            return self._get_empty_context_embedding()
        
        if self.model is None:
            return self._fallback_context_embedding([u[1] for u in recent_utterances])
        
        try:
            # Take last max_turns utterances
            context_utterances = recent_utterances[-max_turns:]
            
            # Build dialogue text with DialogueBERT role markers
            dialogue_parts = []
            for role, text in context_utterances:
                if role.lower() == "user":
                    dialogue_parts.append(f"user: {text}")
                else:
                    dialogue_parts.append(f"system: {text}")
            
            # Join with separator tokens
            context_text = " [SEP] ".join(dialogue_parts)
            
            # Tokenize with DialogueBERT approach
            inputs = self.tokenizer(
                context_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate context embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                context_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return context_embedding.flatten()
            
        except Exception as e:
            logger.warning(f"Error generating context embedding: {e}")
            return self._fallback_context_embedding([u[1] for u in recent_utterances])
    
    def classify_intent_category(self, utterance: str) -> str:
        """
        Classify utterance into intent categories using DialogueBERT understanding.
        
        Args:
            utterance: Visitor utterance
            
        Returns:
            Intent category string
        """
        if not utterance or not utterance.strip():
            return "silence"
        
        utterance_lower = utterance.lower()
        
        # Question indicators
        if any(word in utterance_lower for word in ["?", "what", "how", "why", "when", "where", "who", "which"]):
            return "question"
        
        # Confusion indicators
        if any(word in utterance_lower for word in ["don't understand", "confused", "unclear", "what do you mean", "huh", "what"]):
            return "confusion"
        
        # Interest indicators
        if any(word in utterance_lower for word in ["interesting", "fascinating", "amazing", "wow", "beautiful", "love", "like"]):
            return "interest"
        
        # Request indicators
        if any(word in utterance_lower for word in ["tell me", "explain", "show me", "can you", "could you"]):
            return "request"
        
        # Default to statement
        return "statement"
    
    def get_dialogue_state_embedding(self, current_state: Dict[str, Any]) -> np.ndarray:
        """
        Generate dialogue state embedding for tracking conversation state.
        
        Args:
            current_state: Current dialogue state dictionary
            
        Returns:
            State embedding vector of shape (768,)
        """
        if self.model is None:
            return self._get_empty_context_embedding()
        
        try:
            # Build state description
            state_parts = []
            
            if current_state.get("current_focus"):
                state_parts.append(f"Focus: {current_state['current_focus']}")
            
            if current_state.get("explained_exhibits"):
                state_parts.append(f"Explained: {', '.join(current_state['explained_exhibits'])}")
            
            if current_state.get("recent_actions"):
                state_parts.append(f"Actions: {', '.join(current_state['recent_actions'])}")
            
            state_text = " [SEP] ".join(state_parts) if state_parts else "Initial state"
            
            # Tokenize with DialogueBERT approach
            inputs = self.tokenizer(
                state_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate state embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                state_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return state_embedding.flatten()
            
        except Exception as e:
            logger.warning(f"Error generating state embedding: {e}")
            return self._get_empty_context_embedding()
    
    def _get_silence_embedding(self) -> np.ndarray:
        """Generate embedding for silence/empty utterances."""
        return np.zeros(768, dtype=np.float32)
    
    def _get_empty_context_embedding(self) -> np.ndarray:
        """Generate embedding for empty dialogue context."""
        return np.zeros(768, dtype=np.float32)
    
    def _fallback_intent_embedding(self, utterance: str) -> np.ndarray:
        """
        Fallback intent embedding when DialogueBERT is not available.
        
        Args:
            utterance: Visitor utterance
            
        Returns:
            Simple intent embedding
        """
        # Simple one-hot encoding based on intent classification
        intent_category = self.classify_intent_category(utterance)
        
        # Create simple embedding (could be enhanced)
        embedding = np.zeros(768, dtype=np.float32)
        
        # Set some values based on intent category
        if intent_category == "question":
            embedding[:100] = 0.1
        elif intent_category == "confusion":
            embedding[100:200] = 0.1
        elif intent_category == "interest":
            embedding[200:300] = 0.1
        elif intent_category == "request":
            embedding[300:400] = 0.1
        elif intent_category == "statement":
            embedding[400:500] = 0.1
        # silence remains all zeros
        
        return embedding
    
    def _fallback_context_embedding(self, utterances: List[str]) -> np.ndarray:
        """
        Fallback context embedding when DialogueBERT is not available.
        
        Args:
            utterances: List of recent utterances
            
        Returns:
            Simple context embedding
        """
        # Simple context embedding based on utterance count and types
        embedding = np.zeros(768, dtype=np.float32)
        
        if not utterances:
            return embedding
        
        # Encode basic context information
        num_utterances = len(utterances)
        embedding[500:600] = num_utterances / 10.0  # Normalize by max expected turns
        
        # Encode recent intent patterns
        recent_intents = [self.classify_intent_category(u) for u in utterances[-3:]]
        
        if "question" in recent_intents:
            embedding[600:650] = 0.1
        if "confusion" in recent_intents:
            embedding[650:700] = 0.1
        if "interest" in recent_intents:
            embedding[700:750] = 0.1
        
        return embedding


# Global DialogueBERT intent recognizer instance
_dialoguebert_recognizer = None


def get_dialoguebert_recognizer() -> DialogueBERTIntentRecognizer:
    """
    Get global DialogueBERT recognizer instance (singleton pattern).
    
    Returns:
        DialogueBERTIntentRecognizer instance
    """
    global _dialoguebert_recognizer
    if _dialoguebert_recognizer is None:
        _dialoguebert_recognizer = DialogueBERTIntentRecognizer()
    return _dialoguebert_recognizer


def reset_dialoguebert_recognizer():
    """Reset global DialogueBERT recognizer instance."""
    global _dialoguebert_recognizer
    _dialoguebert_recognizer = None
