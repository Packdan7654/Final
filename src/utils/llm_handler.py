"""
LLM Handler for Museum Dialogue Generation

This module provides a flexible interface for using LLMs. It supports multiple backends:

1. Groq API (recommended): Fast inference with Llama models
2. Hugging Face: Local models (Phi-2, TinyLLaMA, Mistral, FLAN-T5, GPT-2)
3. Mistral API: Free tier available

Default: Groq API with Llama 3.1 8B (fast and efficient)
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMBackend(Enum):
    """Available LLM backends."""
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    MISTRAL_API = "mistral_api"


class LLMCriticalError(Exception):
    """Raised when LLM encounters a critical, unrecoverable error (e.g., spend limit, auth failure)."""
    pass


class FreeLLMHandler:
    """
    Handler for LLM inference.
    
    This provides a unified interface for different LLM backends.
    Default: Groq API with Llama 3.1 8B for fast, efficient inference.
    
    Available Groq models:
    - llama-3.1-8b: Default, fast and cost-effective (8B)
    - llama-3.1: Higher quality (70B)
    - llama-3.3: Latest high quality (70B)
    
    For HuggingFace (local models):
    - phi2, tinyllama, neural-chat, mistral
    """
    
    def __init__(
        self,
        backend: str = "groq",
        model_name: str = "llama-3.1-8b",
        temperature: float = 0.7,
        max_tokens: int = 250,
        device: Optional[str] = None
    ):
        """
        Initialize LLM handler.
        
        Args:
            backend: LLM backend ('groq', 'huggingface', 'mistral_api')
            model_name: Model name (default: 'llama-3.1-8b' for Groq)
                       Options: llama-3.1-8b, llama-3.3, llama-3.1 (Groq)
                       Or local models: phi2, tinyllama, neural-chat, mistral (HuggingFace)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device for local models ('cpu', 'cuda')
        """
        self.backend = LLMBackend(backend.lower())
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if self._has_cuda() else 'cpu')
        
        # Check for fast mode
        import os
        self.fast_mode = os.environ.get('HRL_FAST_MODE') == '1'
        
        # Critical error tracking
        self.critical_error_count = 0
        self.max_critical_errors = 3  # Allow a few errors before failing
        self.last_critical_error = None
        
        if not self.fast_mode:
            # Initialize backend normally
            self._initialize_backend()
        else:
            print("[FAST MODE] Skipping LLM backend initialization")
        
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_backend(self):
        """Initialize the selected LLM backend."""
        if self.backend == LLMBackend.GROQ:
            self._initialize_groq()
        elif self.backend == LLMBackend.HUGGINGFACE:
            self._initialize_huggingface()
        elif self.backend == LLMBackend.MISTRAL_API:
            self._initialize_mistral_api()
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face backend."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading Hugging Face model: {self.model_name}...")
            
            # Map common names to HF model IDs
            model_map = {
                "phi2": "microsoft/phi-2",
                "phi": "microsoft/phi-2",
                "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "neural-chat": "Intel/neural-chat-7b-v3-1",
                "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
                "flan-t5": "google/flan-t5-base",
                "gpt2": "gpt2",
                "gpt2-medium": "gpt2-medium"
            }
            
            model_id = model_map.get(self.model_name.lower(), self.model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None
            )
            
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
            
            print(f"[OK] Loaded {model_id} on {self.device}")
            self.hf_available = True
            
        except ImportError:
            print("[WARN] transformers not installed: pip install transformers torch")
            self.hf_available = False
        except Exception as e:
            print(f"[WARN] Error loading Hugging Face model: {e}")
            self.hf_available = False
    
    def _initialize_mistral_api(self):
        """Initialize Mistral API backend."""
        try:
            if "MISTRAL_API_KEY" not in os.environ:
                print("[WARN] MISTRAL_API_KEY not set in environment")
                self.mistral_api_available = False
            else:
                from mistralai.client import MistralClient
                self.mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
                print("[OK] Mistral API client initialized")
                self.mistral_api_available = True
        except ImportError:
            print("[WARN] mistralai not installed: pip install mistralai")
    
    def _initialize_groq(self):
        """Initialize Groq API backend."""
        try:
            # Try to get API key from environment first
            api_key = os.environ.get("GROQ_API_KEY")
            
            # If not in environment, try to read from key.txt
            if not api_key:
                key_file = Path(__file__).parent.parent.parent / "key.txt"
                if key_file.exists():
                    try:
                        api_key = key_file.read_text().strip()
                        print(f"[OK] Loaded Groq API key from key.txt")
                        # Set it in environment for future use
                        os.environ["GROQ_API_KEY"] = api_key
                    except Exception as e:
                        print(f"[WARN] Failed to read key.txt: {e}")
                        api_key = None
            
            if not api_key:
                print("[WARN] GROQ_API_KEY not set in environment and key.txt not found")
                print("  Get free API key: https://console.groq.com/")
                print("  Or create key.txt in project root with your API key")
                self.groq_available = False
            else:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                print(f"[OK] Groq API client initialized")
                print(f"  Using model: {self.model_name}")
                self.groq_available = True
        except ImportError:
            print("[WARN] groq not installed: pip install groq")
            self.groq_available = False
        except Exception as e:
            print(f"[WARN] Error initializing Groq: {e}")
            self.groq_available = False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using the configured LLM backend.
        
        Args:
            prompt: User/input prompt
            system_prompt: System prompt (context)
            
        Returns:
            Generated text
        """
        # In fast mode, always use fallback
        if self.fast_mode:
            return self._fallback_response(prompt)
        
        if self.backend == LLMBackend.GROQ:
            return self._generate_groq(prompt, system_prompt)
        elif self.backend == LLMBackend.HUGGINGFACE:
            return self._generate_huggingface(prompt, system_prompt)
        elif self.backend == LLMBackend.MISTRAL_API:
            return self._generate_mistral_api(prompt, system_prompt)
        else:
            return self._fallback_response(prompt)
    
    def _generate_huggingface(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Hugging Face."""
        if not self.hf_available:
            return self._fallback_response(prompt)
        
        try:
            # Format prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response (remove prompt)
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"[WARN] Hugging Face generation error: {e}")
            return self._fallback_response(prompt)
    
    def _generate_mistral_api(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Mistral API."""
        if not self.mistral_client:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.mistral_client.chat(
                model=self.model_name or "mistral-small",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[WARN] Mistral API error: {e}")
            return self._fallback_response(prompt)
    
    def _generate_groq(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Groq API with critical error detection."""
        if not self.groq_available:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Map common model names to Groq model IDs (updated for current models)
            model_map = {
                "mistral": "llama-3.3-70b-versatile",  # Use Llama 3.3 as replacement
                "mixtral": "llama-3.3-70b-versatile",
                "llama": "llama-3.3-70b-versatile",
                "llama2": "llama-3.1-70b-versatile",
                "llama3": "llama-3.3-70b-versatile",
                "llama-3.3": "llama-3.3-70b-versatile",
                "llama-3.1": "llama-3.1-70b-versatile",
                "llama-3.1-8b": "llama-3.1-8b-instant",  # Cheaper 8B model!
            }
            groq_model = model_map.get(self.model_name.lower(), self.model_name)
            
            response = self.groq_client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Reset error count on success
            self.critical_error_count = 0
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e)
            print(f"[WARN] Groq API error: {e}")
            
            # Detect CRITICAL errors that should stop training
            critical_error_keywords = [
                'spend_limit_reached',
                'spend alert threshold',
                'insufficient_quota',
                'rate_limit_exceeded',
                'authentication_error',
                'invalid_api_key',
                'account_deactivated'
            ]
            
            is_critical = any(keyword in error_str.lower() for keyword in critical_error_keywords)
            
            if is_critical:
                self.critical_error_count += 1
                self.last_critical_error = error_str
                
                print(f"\n{'='*80}")
                print(f"CRITICAL LLM ERROR DETECTED ({self.critical_error_count}/{self.max_critical_errors})")
                print(f"{'='*80}")
                print(f"Error: {error_str[:200]}")
                
                if self.critical_error_count >= self.max_critical_errors:
                    print(f"\nCritical error threshold reached!")
                    print(f"Training will be stopped to save the model.")
                    print(f"{'='*80}\n")
                    raise LLMCriticalError(f"Groq API critical error: {error_str}")
            
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback template-based response when LLM is unavailable.
        
        Provides museum dialogue responses based on prompt keywords.
        In fast mode, includes fact IDs for testing reward calculations.
        """
        prompt_lower = prompt.lower()
        
        # Extract action type from prompt
        if "explain" in prompt_lower or "tell" in prompt_lower or "fact" in prompt_lower:
            # For Explain actions, include fact IDs so novelty rewards work
            import random
            fact_ids = ["[KC_001]", "[KC_002]", "[KC_003]", "[KC_004]", "[KC_005]", "[KC_006]"]
            selected_facts = random.sample(fact_ids, 2)
            return f"This exhibit is fascinating. {selected_facts[0]} It represents wealth and status. {selected_facts[1]} The craftsmanship is remarkable."
        
        elif "ask" in prompt_lower and "opinion" in prompt_lower:
            return "What do you think about this piece? Does the craftsmanship appeal to you?"
        
        elif "ask" in prompt_lower and "memory" in prompt_lower:
            return "Do you remember the details we discussed about the previous exhibit?"
        
        elif "transition" in prompt_lower or "move" in prompt_lower:
            return "Shall we explore another fascinating piece nearby?"
        
        elif "conclude" in prompt_lower or "wrap" in prompt_lower:
            return "Thank you for visiting! I hope you've enjoyed learning about these remarkable exhibits."
        
        else:
            generic = [
                "That's an insightful observation about this exhibit.",
                "This piece has a rich history worth exploring.",
                "The cultural significance here is quite remarkable.",
                "There's much to appreciate in the details of this work.",
            ]
            import random
            return random.choice(generic)


# Global LLM handler instance
_llm_handler: Optional[FreeLLMHandler] = None


def get_llm_handler(
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> FreeLLMHandler:
    """
    Get global LLM handler instance (singleton).
    
    Args:
        backend: Override backend ('groq', 'huggingface', 'mistral_api')
        model_name: Override model name
        **kwargs: Additional arguments for FreeLLMHandler
        
    Returns:
        FreeLLMHandler instance
    """
    global _llm_handler
    
    # Check environment variables for configuration
    if backend is None:
        backend = os.environ.get("HRL_LLM_BACKEND", "groq")
    if model_name is None:
        model_name = os.environ.get("HRL_LLM_MODEL", "llama-3.1-8b")
    
    if _llm_handler is None:
        _llm_handler = FreeLLMHandler(
            backend=backend,
            model_name=model_name,
            **kwargs
        )
    
    return _llm_handler


def reset_llm_handler():
    """Reset global LLM handler instance."""
    global _llm_handler
    _llm_handler = None

