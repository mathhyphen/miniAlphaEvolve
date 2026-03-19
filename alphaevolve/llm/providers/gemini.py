"""Google Gemini API client for AlphaEvolve."""

import logging
import time
from typing import Any, Dict, Optional, AsyncIterator, TYPE_CHECKING

from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.config import LLMConfig

logger = logging.getLogger(__name__)

# For type checking only
if TYPE_CHECKING:
    try:
        import google.generativeai as genai
    except ImportError:
        pass

# Try to import google.generativeai, provide helpful error if not installed
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai package not installed. "
        "Install with: pip install google-generativeai"
    )


class GeminiClient(BaseLLMClient):
    """Google Gemini API client.

    Supports Gemini family models:
    - gemini-2.0-flash (fast, efficient)
    - gemini-1.5-pro (highest quality)
    - gemini-1.5-flash (fast)

    Features:
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Rate limit handling
    """

    # Available Gemini models
    MODELS = {
        "gemini-2.5-flash": {"context": 1048576, "output": 8192},
        "gemini-2.5-pro": {"context": 2097152, "output": 8192},
        "gemini-2.0-flash-lite": {"context": 1048576, "output": 8192},
    }

    def __init__(self, config: LLMConfig) -> None:
        """Initialize Gemini client.

        Args:
            config: LLM configuration
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

        self.config = config
        self._api_key = config.api_key
        self._client = None
        self._model = None
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
        return "gemini"

    @property
    def client(self):
        """Lazy-initialized Gemini client."""
        if self._client is None:
            api_key = self._api_key
            if not api_key:
                import os
                api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. "
                    "Set it in environment or pass api_key in config."
                )
            genai.configure(api_key=api_key)
            self._client = genai
            self._model = genai.GenerativeModel(self.config.model)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate text using Gemini API.

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

        # Build generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Build messages with system prompt
        messages = prompt
        if system_prompt:
            messages = f"{system_prompt}\n\n{prompt}"

        # Add retry logic
        last_error = None
        for attempt in range(self._max_retries):
            try:
                # Generate response
                response = self.model.generate_content(
                    messages,
                    generation_config=generation_config,
                    **kwargs
                )

                # Extract usage statistics
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": usage_metadata.prompt_token_count if usage_metadata else 0,
                    "completion_tokens": usage_metadata.candidates_token_count if usage_metadata else 0,
                    "total_tokens": usage_metadata.total_token_count if usage_metadata else 0,
                }

                # Create response
                llm_response = LLMResponse(
                    content=response.text if response.text else "",
                    model=self.config.model,
                    usage=usage,
                    finish_reason="stop",
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
                logger.warning(f"Gemini API error, retrying in {delay:.1f}s: {e}")
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
        messages = prompt
        if system_prompt:
            messages = f"{system_prompt}\n\n{prompt}"

        response = self.model.generate_content(messages, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

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
        """List available Gemini models."""
        return list(self.MODELS.keys())

    @property
    def model(self):
        """Get the generative model instance."""
        if self._model is None:
            _ = self.client  # Initialize client first
        return self._model
