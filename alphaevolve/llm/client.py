"""Base LLM client interface for AlphaEvolve."""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator


@dataclass
class LLMResponse:
    """Standardized LLM response format.

    Args:
        content: Generated text content
        model: Model that generated the response
        usage: Token usage statistics
        finish_reason: Reason for completion
        raw_response: Original API response (provider-specific)
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate text completion.

        Args:
            prompt: User prompt for generation
            system_prompt: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with generated content
        """
        pass

    @abc.abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream text generation.

        Args:
            prompt: User prompt for generation
            system_prompt: Optional system instruction
            **kwargs: Provider-specific arguments

        Yields:
            Chunks of generated text
        """
        pass

    @abc.abstractmethod
    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        pass

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        pass
