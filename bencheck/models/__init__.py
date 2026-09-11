"""Model implementations for BenCheck."""

from .dummy import DummyModel

__all__ = ["DummyModel"]

# Grammar detector model (optional)
try:
    from .gector_grammar import GECToRGrammarModel

    __all__.append("GECToRGrammarModel")
except ImportError:
    pass

# vLLM model (optional)
try:
    from .vllm_model import VLLMModel

    __all__.append("VLLMModel")
except ImportError:
    pass

# Gemini model (optional)
try:
    from .gemini_model import GeminiModel

    __all__.append("GeminiModel")
except ImportError:
    pass

# OpenAI-compatible model (optional)
try:
    from .openai_model import OpenAIModel

    __all__.append("OpenAIModel")
except ImportError:
    pass
