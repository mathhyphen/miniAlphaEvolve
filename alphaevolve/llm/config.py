"""LLM configuration for AlphaEvolve."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM provider.

    Args:
        provider: Provider name (e.g., "anthropic", "google", "bailian")
        model: Model name (e.g., "claude-3-5-sonnet-20241022", "kimi-k2-0711-preview")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        api_key: API key (can be None if set in environment)
        timeout: Request timeout in seconds
        base_url: Optional base URL for API endpoint (for OpenAI-compatible APIs)
    """
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    timeout: float = 60.0
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
