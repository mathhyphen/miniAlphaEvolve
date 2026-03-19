"""Aliyun Bailian (Kimi) API client for AlphaEvolve.

This module provides a client for Aliyun Bailian's Kimi models,
which are compatible with the OpenAI API format.

Bailian Endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
"""

import logging
import time
from typing import Any, Dict, Optional, AsyncIterator, TYPE_CHECKING

from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.config import LLMConfig

logger = logging.getLogger(__name__)

# For type checking only
if TYPE_CHECKING:
    try:
        from openai import OpenAI
    except ImportError:
        pass

# Try to import openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning(
        "openai package not installed. "
        "Install with: pip install openai"
    )


class BailianClient(BaseLLMClient):
    """Aliyun Bailian (Kimi) API client.

    Supports Kimi models via OpenAI-compatible API:
    - kimi-k2-0711-preview (Kimi 2.5)
    - kimi-latest

    Features:
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Rate limit handling
    """

    # Kimi models available on Aliyun Bailian
    MODELS = {
        "kimi-k2-0711-preview": {"context": 131072, "output": 8192},  # Kimi 2.5
        "kimi-latest": {"context": 131072, "output": 8192},
    }

    # Aliyun Bailian OpenAI-compatible endpoint
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Bailian client.

        Args:
            config: LLM configuration
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required. "
                "Install with: pip install openai"
            )

        self.config = config
        self._api_key = config.api_key
        self._base_url = config.base_url or self.BASE_URL
        self._client = None
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
        return "bailian"

    @property
    def client(self) -> "OpenAI":
        """Lazy-initialized OpenAI client for Bailian."""
        if self._client is None:
            api_key = self._api_key
            if not api_key:
                import os
                # Try multiple environment variable names for compatibility
                api_key = (
                    os.environ.get("CODING_PLAN_API_KEY")
                    or os.environ.get("ALI_BAILIAN_API_KEY")
                    or os.environ.get("ANTHROPIC_API_KEY")
                )
            if not api_key:
                raise ValueError(
                    "API key not set. Set CODING_PLAN_API_KEY or ALI_BAILIAN_API_KEY in .env. "
                    "Your Coding Plan API key should start with 'sk-sp-'."
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self._base_url,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate text using Bailian Kimi API.

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

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Add retry logic
        last_error = None
        for attempt in range(self._max_retries):
            try:
                # Generate response
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                # Extract usage statistics
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                # Create response
                llm_response = LLMResponse(
                    content=response.choices[0].message.content if response.choices and response.choices[0].message.content else "",
                    model=self.config.model,
                    usage=usage,
                    finish_reason=response.choices[0].finish_reason if response.choices else "stop",
                    raw_response=response,
                )

                # Update internal usage tracking
                self._usage["total_requests"] += 1
                self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self._usage["total_tokens"] += usage.get("total_tokens", 0)

                self._retry_count = 0  # Reset on success
                return llm_response

            except Exception as e:
                last_error = e
                self._retry_count += 1
                # Exponential backoff with jitter
                delay = self._base_delay * (2 ** self._retry_count) + random.uniform(0, 1)
                logger.warning(f"Bailian API error, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)

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
        """Stream text generation.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            **kwargs: Additional API arguments

        Yields:
            Chunks of generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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

    def list_models(self) -> list:
        """List available Kimi models on Bailian."""
        return list(self.MODELS.keys())
