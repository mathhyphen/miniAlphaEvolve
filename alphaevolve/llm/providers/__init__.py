"""LLM provider registry and factory."""

from typing import Dict, Type, Any
from alphaevolve.llm.client import BaseLLMClient
from alphaevolve.llm.config import LLMConfig

# Registry of LLM providers
LLM_FACTORY: Dict[str, Type[BaseLLMClient]] = {}


def register_llm(name: str) -> Any:
    """Decorator to register an LLM provider.

    Usage:
        @register_llm("anthropic")
        class AnthropicClient(BaseLLMClient):
            pass
    """
    def decorator(cls: Type[BaseLLMClient]) -> Type[BaseLLMClient]:
        LLM_FACTORY[name] = cls
        return cls
    return decorator


def LLMProviderFactory(provider: str, config: LLMConfig) -> BaseLLMClient:
    """Factory function to create LLM clients.

    Args:
        provider: Provider name (e.g., "anthropic", "google")
        config: LLM configuration

    Returns:
        Configured LLM client instance

    Raises:
        ValueError: If provider is not registered
    """
    client_class = LLM_FACTORY.get(provider)
    if not client_class:
        available = ", ".join(LLM_FACTORY.keys())
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Available providers: {available}"
        )
    return client_class(config)


__all__ = [
    "LLM_FACTORY",
    "register_llm",
    "LLMProviderFactory",
    "AnthropicClient",
    "MiniMaxClient",
]

# Import providers to register them
# Note: Import at end to avoid circular imports
try:
    from alphaevolve.llm.providers.anthropic import AnthropicClient  # noqa: F401
except ImportError:
    pass

try:
    from alphaevolve.llm.providers.minimax import MiniMaxClient  # noqa: F401
except ImportError:
    pass
