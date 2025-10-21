"""
Centralized LLM Configuration

This module provides a single place to configure which LLM models to use
for the agent and the user simulator. Easy switching between models for
testing and evaluation.

Available Models:
- orca-mini (3.3B): Fastest, good quality
- mistral:small (7B): Fast Mistral variant
- mistral:latest (7B): Standard Mistral
- neural-chat (7B): Good for dialogue
"""

import os


# ============================================================================
# CENTRALIZED MODEL CONFIGURATION - EDIT HERE TO CHANGE MODELS
# ============================================================================

class LLMConfig:
    """Centralized LLM configuration for easy switching."""
    
    # ========================================================================
    # AGENT LLM (Used for museum guide dialogue generation in training)
    # ========================================================================
    AGENT_LLM_MODEL = "mistral"  # High quality for comparison
    
    # Model options:
    # - "orca-mini"        : Ultra-fast (3.3B), lower quality
    # - "mistral-small3.1:latest quality
    # - "mistral:latest"   : Balanced (7B), high quality
    # - "neural-chat"      : Good for dialogue (7B)
    
    # Temperature for agent (lower = more deterministic)
    AGENT_TEMPERATURE = 0.3  # Low for consistent, factual responses
    AGENT_MAX_TOKENS = 300
    
    # ========================================================================
    # SIMULATOR/USER LLM (Used for user response generation in simulator)
    # ========================================================================
    SIMULATOR_LLM_MODEL = "mistral"  # High quality for comparison
    
    # Model options (same as above)
    # Use a faster model here since user responses are simpler
    
    # Temperature for simulator (lower = more consistent)
    SIMULATOR_TEMPERATURE = 0.6
    SIMULATOR_MAX_TOKENS = 150
    
    # ========================================================================
    # BACKEND CONFIGURATION
    # ========================================================================
    LLM_BACKEND = "ollama"  # Options: "ollama", "huggingface", "mistral_api"
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # ========================================================================
    # PRESET CONFIGURATIONS (Copy and paste to switch)
    # ========================================================================
    
    @classmethod
    def preset_fast_testing(cls):
        """Fastest setup - good for quick iteration"""
        cls.AGENT_LLM_MODEL = "orca-mini"
        cls.SIMULATOR_LLM_MODEL = "orca-mini"
        print("[CONFIG] Preset: FAST TESTING (both orca-mini)")
    
    @classmethod
    def preset_balanced(cls):
        """Balanced setup - good for development"""
        cls.AGENT_LLM_MODEL = "mistral:small"
        cls.SIMULATOR_LLM_MODEL = "orca-mini"
        print("[CONFIG] Preset: BALANCED (agent=mistral:small, user=orca-mini)")
    
    @classmethod
    def preset_high_quality(cls):
        """High quality setup - for evaluation"""
        cls.AGENT_LLM_MODEL = "mistral:latest"
        cls.SIMULATOR_LLM_MODEL = "neural-chat"
        print("[CONFIG] Preset: HIGH QUALITY (agent=mistral:latest, user=neural-chat)")
    
    @classmethod
    def preset_neural_chat(cls):
        """Neural chat optimized setup - good for dialogue"""
        cls.AGENT_LLM_MODEL = "neural-chat"
        cls.SIMULATOR_LLM_MODEL = "neural-chat"
        print("[CONFIG] Preset: NEURAL CHAT (both neural-chat)")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 70)
        print("CURRENT LLM CONFIGURATION")
        print("=" * 70)
        print(f"Agent LLM:      {cls.AGENT_LLM_MODEL}")
        print(f"  Temperature:  {cls.AGENT_TEMPERATURE}")
        print(f"  Max tokens:   {cls.AGENT_MAX_TOKENS}")
        print()
        print(f"Simulator LLM:  {cls.SIMULATOR_LLM_MODEL}")
        print(f"  Temperature:  {cls.SIMULATOR_TEMPERATURE}")
        print(f"  Max tokens:   {cls.SIMULATOR_MAX_TOKENS}")
        print()
        print(f"Backend:        {cls.LLM_BACKEND}")
        print(f"Ollama Host:    {cls.OLLAMA_HOST}")
        print("=" * 70 + "\n")


# ============================================================================
# ENVIRONMENT VARIABLE OVERRIDES (Optional)
# ============================================================================
# You can override via environment variables if needed:

if os.environ.get("HRL_AGENT_LLM"):
    LLMConfig.AGENT_LLM_MODEL = os.environ["HRL_AGENT_LLM"]

if os.environ.get("HRL_SIMULATOR_LLM"):
    LLMConfig.SIMULATOR_LLM_MODEL = os.environ["HRL_SIMULATOR_LLM"]

if os.environ.get("HRL_LLM_BACKEND"):
    LLMConfig.LLM_BACKEND = os.environ["HRL_LLM_BACKEND"]


# ============================================================================
# GLOBAL HANDLER INSTANCES
# ============================================================================

_agent_llm_handler = None
_simulator_llm_handler = None


def get_agent_llm():
    """Get or create the agent LLM handler"""
    global _agent_llm_handler
    if _agent_llm_handler is None:
        from src.utils.llm_handler import FreeLLMHandler
        _agent_llm_handler = FreeLLMHandler(
            backend=LLMConfig.LLM_BACKEND,
            model_name=LLMConfig.AGENT_LLM_MODEL,
            temperature=LLMConfig.AGENT_TEMPERATURE,
            max_tokens=LLMConfig.AGENT_MAX_TOKENS
        )
    return _agent_llm_handler


def get_simulator_llm():
    """Get or create the simulator LLM handler"""
    global _simulator_llm_handler
    if _simulator_llm_handler is None:
        from src.utils.llm_handler import FreeLLMHandler
        _simulator_llm_handler = FreeLLMHandler(
            backend=LLMConfig.LLM_BACKEND,
            model_name=LLMConfig.SIMULATOR_LLM_MODEL,
            temperature=LLMConfig.SIMULATOR_TEMPERATURE,
            max_tokens=LLMConfig.SIMULATOR_MAX_TOKENS
        )
    return _simulator_llm_handler


def reset_llm_handlers():
    """Reset cached LLM handlers (useful for testing)"""
    global _agent_llm_handler, _simulator_llm_handler
    _agent_llm_handler = None
    _simulator_llm_handler = None


if __name__ == "__main__":
    # Print current configuration
    LLMConfig.print_config()
    
    # Try different presets
    print("\nAvailable presets:")
    print("  - LLMConfig.preset_fast_testing()")
    print("  - LLMConfig.preset_balanced()")
    print("  - LLMConfig.preset_high_quality()")
    print("  - LLMConfig.preset_neural_chat()")
