"""Anthropic Claude API client for AlphaEvolve."""

import logging
import time
from typing import Any, Dict, Optional, AsyncIterator, TYPE_CHECKING

from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.config import LLMConfig

logger = logging.getLogger(__name__)

# For type checking only
if TYPE_CHECKING:
    try:
        from anthropic import Anthropic
    except ImportError:
        pass

# Try to import anthropic, provide helpful error if not installed
try:
    import anthropic
    from anthropic import Anthropic, RateLimitError, APIError  # noqa: F401
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning(
        "anthropic package not installed. "
        "Install with: pip install anthropic"
    )


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client.

    Supports Claude 3 family models:
    - claude-3-5-sonnet-20241022 (recommended for code)
    - claude-3-opus-20240229 (highest quality)
    - claude-3-haiku-20240307 (fastest)

    Features:
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Rate limit handling
    """

    # Available Claude models
    MODELS = {
        "claude-3-5-sonnet-20241022": {"context": 200000, "output": 8192},
        "claude-3-opus-20240229": {"context": 200000, "output": 4096},
        "claude-3-haiku-20240307": {"context": 200000, "output": 4096},
    }

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Anthropic client.

        Args:
            config: LLM configuration
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.config = config
        self._api_key = config.api_key
        self._client: Optional[Anthropic] = None
        self._usage = {
            "total_requests": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self._retry_count = 0
        self._max_retries = 3
        self._base_delay = 1.0  # seconds

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def client(self) -> "Anthropic":
        """Lazy-initialized Anthropic client."""
        if self._client is None:
            api_key = self._api_key
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it in environment or pass api_key in config."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate text using Claude API.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional API arguments

        Returns:
            LLMResponse with generated content
        """
        import random

        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Build API arguments
        api_args = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            api_args["system"] = system_prompt

        # Add retry logic
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self.client.messages.create(**api_args)

                # Update usage statistics
                self._update_usage(response.usage)

                # Create response
                llm_response = LLMResponse(
                    content=response.content[0].text if response.content else "",
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    },
                    finish_reason=response.stop_reason,
                    raw_response=response,
                )

                self._usage["total_requests"] += 1
                self._retry_count = 0  # Reset on success
                return llm_response

            except RateLimitError as e:
                last_error = e
                self._retry_count += 1
                # Exponential backoff with jitter
                delay = self._base_delay * (2 ** self._retry_count) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)

            except APIError as e:
                last_error = e
                logger.error(f"API error: {e}")
                break

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                break

        # All retries exhausted
        error_msg = f"Failed after {self._max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream text generation (not yet implemented).

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            **kwargs: Additional API arguments

        Yields:
            Chunks of generated text
        """
        # Note: Streaming requires async client
        # This is a placeholder for future implementation
        response = self.generate(prompt, system_prompt, **kwargs)
        yield response.content

    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage.copy()

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._usage = {
            "total_requests": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _update_usage(self, usage: Any) -> None:
        """Update internal usage tracking."""
        self._usage["prompt_tokens"] += usage.input_tokens
        self._usage["completion_tokens"] += usage.output_tokens
        self._usage["total_tokens"] += usage.input_tokens + usage.output_tokens

    def list_models(self) -> list:
        """List available Claude models."""
        return list(self.MODELS.keys())
