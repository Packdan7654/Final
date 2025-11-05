"""
Centralized LLM Configuration

This module provides a single place to configure which LLM models to use
for the agent and the user simulator. Easy switching between models for
testing and evaluation.

Available Models (Groq API):
- llama-3.1-8b: Fast and cost-effective (8B) - recommended
- llama-3.1: Higher quality (70B)
- llama-3.3: Latest high quality (70B)
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
    AGENT_LLM_MODEL = "llama-3.1-8b"  # Groq: Fast and cost-effective (8B model)
    
    # Model options (Groq):
    # - "llama-3.1-8b"     : Fast and cheap (8B) - recommended for training
    # - "llama-3.1"        : Higher quality (70B) - more expensive
    # - "llama-3.3"        : Latest high quality (70B) - most expensive
    
    # Temperature for agent (lower = more deterministic)
    AGENT_TEMPERATURE = 0.3  # Low for consistent, factual responses
    AGENT_MAX_TOKENS = 300
    
    # ========================================================================
    # SIMULATOR/USER LLM (Used for user response generation in simulator)
    # ========================================================================
    SIMULATOR_LLM_MODEL = "llama-3.1-8b"  # Groq: Fast and cost-effective
    
    # Model options (same as above)
    # Use a faster/cheaper model here since user responses are simpler
    
    # Temperature for simulator (lower = more consistent)
    SIMULATOR_TEMPERATURE = 0.6
    SIMULATOR_MAX_TOKENS = 150
    
    # ========================================================================
    # BACKEND CONFIGURATION
    # ========================================================================
    LLM_BACKEND = "groq"  # Options: "groq", "huggingface", "mistral_api"
    
    # ========================================================================
    # PRESET CONFIGURATIONS (Copy and paste to switch)
    # ========================================================================
    
    @classmethod
    def preset_fast_testing(cls):
        """Fastest setup - good for quick iteration"""
        cls.AGENT_LLM_MODEL = "llama-3.1-8b"
        cls.SIMULATOR_LLM_MODEL = "llama-3.1-8b"
        print("[CONFIG] Preset: FAST TESTING (both llama-3.1-8b)")
    
    @classmethod
    def preset_balanced(cls):
        """Balanced setup - good for development"""
        cls.AGENT_LLM_MODEL = "llama-3.1"
        cls.SIMULATOR_LLM_MODEL = "llama-3.1-8b"
        print("[CONFIG] Preset: BALANCED (agent=llama-3.1, user=llama-3.1-8b)")
    
    @classmethod
    def preset_high_quality(cls):
        """High quality setup - for evaluation"""
        cls.AGENT_LLM_MODEL = "llama-3.3"
        cls.SIMULATOR_LLM_MODEL = "llama-3.1"
        print("[CONFIG] Preset: HIGH QUALITY (agent=llama-3.3, user=llama-3.1)")
    
    @classmethod
    def preset_neural_chat(cls):
        """Balanced dialogue setup - good for dialogue"""
        cls.AGENT_LLM_MODEL = "llama-3.1"
        cls.SIMULATOR_LLM_MODEL = "llama-3.1"
        print("[CONFIG] Preset: BALANCED DIALOGUE (both llama-3.1)")
    
    @classmethod
    def preset_groq_fast(cls):
        """Groq API setup - EXTREMELY FAST (10-20x faster than local)"""
        cls.LLM_BACKEND = "groq"
        cls.AGENT_LLM_MODEL = "llama-3.3"  # Llama 3.3 70B on Groq
        cls.SIMULATOR_LLM_MODEL = "llama-3.3"
        print("[CONFIG] Preset: GROQ FAST (both llama-3.3 via Groq API)")
        print("  ⚡ Extremely fast inference!")
        print("  📝 Set GROQ_API_KEY env variable")
    
    @classmethod
    def preset_groq_llama(cls):
        """Groq API with Llama 3.1 - Very fast and high quality"""
        cls.LLM_BACKEND = "groq"
        cls.AGENT_LLM_MODEL = "llama-3.1"  # Llama 3.1 70B on Groq
        cls.SIMULATOR_LLM_MODEL = "llama-3.1"
        print("[CONFIG] Preset: GROQ LLAMA (both llama-3.1 via Groq API)")
        print("  ⚡ Extremely fast with excellent quality!")
        print("  📝 Set GROQ_API_KEY env variable")
    
    @classmethod
    def preset_groq_cheap(cls):
        """Groq API with Llama 3.1 8B - CHEAPEST! 5-10x cheaper than 70B"""
        cls.LLM_BACKEND = "groq"
        cls.AGENT_LLM_MODEL = "llama-3.1-8b"  # Llama 3.1 8B - Much cheaper!
        cls.SIMULATOR_LLM_MODEL = "llama-3.1-8b"
        print("[CONFIG] Preset: GROQ CHEAP (both llama-3.1-8b via Groq API)")
        print("  💰 5-10x CHEAPER than 70B models!")
        print("  ⚡ Still very fast!")
        print("  📊 500 episodes: ~$1-2 instead of ~$10")
        print("  📝 Set GROQ_API_KEY env variable")
    
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

# Single variable to set both agent and simulator model
if os.environ.get("HRL_LLM_MODEL"):
    LLMConfig.AGENT_LLM_MODEL = os.environ["HRL_LLM_MODEL"]
    LLMConfig.SIMULATOR_LLM_MODEL = os.environ["HRL_LLM_MODEL"]


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
    print("  - LLMConfig.preset_groq_fast()  ⚡ FASTEST!")
    print("  - LLMConfig.preset_groq_llama() ⚡ FAST + HIGH QUALITY!")
