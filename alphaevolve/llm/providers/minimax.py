"""MiniMax API client for AlphaEvolve."""

import logging
import time
import random
from typing import Any, Dict, Optional, AsyncIterator

from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.config import LLMConfig
from alphaevolve.llm.providers import register_llm

logger = logging.getLogger(__name__)


@register_llm("minimax")
class MiniMaxClient(BaseLLMClient):
    """MiniMax API client.

    Supports MiniMax-M2 model through OpenAI-compatible API.

    Features:
    - Automatic retry with exponential backoff
    - Token usage tracking
    """

    # Available models
    MODELS = {
        "MiniMax-M2.7": {"context": 100000, "output": 8192},
    }

    def __init__(self, config: LLMConfig) -> None:
        """Initialize MiniMax client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._api_key = config.api_key
        self._base_url = config.base_url or "https://api.minimax.chat/v1"
        self._client = None
        self._usage = {
            "total_requests": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self._retry_count = 0
        self._max_retries = 3
        self._base_delay = 1.0

    @property
    def name(self) -> str:
        return "minimax"

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for MiniMax. Install with: pip install openai"
                )

            api_key = self._api_key
            if not api_key:
                import os
                api_key = os.environ.get("MINIMAX_API_KEY")

            if not api_key:
                raise ValueError(
                    "MINIMAX_API_KEY not set. "
                    "Set it in environment or pass api_key in config."
                )

            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self._base_url
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
        """Generate text using MiniMax API.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional API arguments

        Returns:
            LLMResponse with generated content
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                # Update usage statistics
                usage = response.usage
                self._usage["prompt_tokens"] += usage.prompt_tokens
                self._usage["completion_tokens"] += usage.completion_tokens
                self._usage["total_tokens"] += usage.prompt_tokens + usage.completion_tokens
                self._usage["total_requests"] += 1

                content = response.choices[0].message.content if response.choices else ""

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    },
                    finish_reason=response.choices[0].finish_reason if response.choices else None,
                    raw_response=response,
                )

            except Exception as e:
                last_error = e
                self._retry_count += 1
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** self._retry_count) + random.uniform(0, 1)
                    logger.warning(f"MiniMax API error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)

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

    def list_models(self) -> list:
        """List available MiniMax models."""
        return list(self.MODELS.keys())
