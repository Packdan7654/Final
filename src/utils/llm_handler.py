"""
Free LLM Handler for Museum Dialogue Generation

This module provides a flexible interface for using free/open-source LLMs
instead of paid APIs. It supports multiple backends:

1. Ollama (recommended): Fast local inference with Phi-2 (default, faster & smaller than Mistral)
2. Hugging Face: Local models (Phi-2, TinyLLaMA, Mistral, FLAN-T5, GPT-2)
3. Mistral API: Free tier available

Default: Ollama with Phi-2 (2.7B parameters - much faster than Mistral-7B)
"""

import os
import json
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMBackend(Enum):
    """Available LLM backends."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    MISTRAL_API = "mistral_api"


class FreeLLMHandler:
    """
    Handler for free/open-source LLM inference.
    
    This provides a unified interface for different LLM backends.
    Default: Phi-2 (2.7B) for fast, efficient inference.
    
    Available fast models:
    - phi2 (2.7B): Default, good quality and speed
    - tinyllama (1.1B): Faster but lower quality  
    - neural-chat (7B): Slightly larger, better quality
    - mistral (7B): Larger, slower but highest quality
    """
    
    def __init__(
        self,
        backend: str = "ollama",
        model_name: str = "phi2",
        temperature: float = 0.7,
        max_tokens: int = 250,
        device: Optional[str] = None
    ):
        """
        Initialize free LLM handler.
        
        Args:
            backend: LLM backend ('ollama', 'huggingface', 'mistral_api')
            model_name: Model name (default: 'phi2' for fast inference)
                       Options: phi2, tinyllama, neural-chat, mistral
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
        if self.backend == LLMBackend.OLLAMA:
            self._initialize_ollama()
        elif self.backend == LLMBackend.HUGGINGFACE:
            self._initialize_huggingface()
        elif self.backend == LLMBackend.MISTRAL_API:
            self._initialize_mistral_api()
    
    def _initialize_ollama(self):
        """Initialize Ollama backend."""
        try:
            import requests
            self.ollama_available = True
            self.ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            
            # Test connection
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    print(f"[OK] Ollama connected at {self.ollama_url}")
                    print(f"  Using model: {self.model_name}")
                else:
                    print(f"[WARN] Ollama connection issue: {response.status_code}")
                    self.ollama_available = False
            except requests.exceptions.RequestException:
                print(f"[WARN] Ollama not running at {self.ollama_url}")
                print("  Install: https://ollama.ai")
                print(f"  Run: ollama pull {self.model_name}")
                self.ollama_available = False
        except ImportError:
            print("[WARN] requests not installed: pip install requests")
            self.ollama_available = False
    
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
        
        if self.backend == LLMBackend.OLLAMA:
            return self._generate_ollama(prompt, system_prompt)
        elif self.backend == LLMBackend.HUGGINGFACE:
            return self._generate_huggingface(prompt, system_prompt)
        elif self.backend == LLMBackend.MISTRAL_API:
            return self._generate_mistral_api(prompt, system_prompt)
        else:
            return self._fallback_response(prompt)
    
    def _generate_ollama(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate using Ollama."""
        if not self.ollama_available:
            return self._fallback_response(prompt)
        
        try:
            import requests
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content'].strip()
            else:
                print(f"[WARN] Ollama error: {response.status_code}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            print(f"[WARN] Ollama generation error: {e}")
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
        backend: Override backend ('ollama', 'huggingface', 'mistral_api')
        model_name: Override model name
        **kwargs: Additional arguments for FreeLLMHandler
        
    Returns:
        FreeLLMHandler instance
    """
    global _llm_handler
    
    # Check environment variables for configuration
    if backend is None:
        backend = os.environ.get("HRL_LLM_BACKEND", "ollama")
    if model_name is None:
        model_name = os.environ.get("HRL_LLM_MODEL", "phi2")
    
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

