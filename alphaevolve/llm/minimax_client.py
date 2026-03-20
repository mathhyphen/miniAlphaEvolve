"""MiniMax API client for AlphaEvolve.

This module provides integration with the MiniMax chat completion API
for generating and optimizing code variants.

Environment Variables:
    MINIMAX_API_KEY: API key for MiniMax API authentication.

"""

import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

API_ENDPOINT = "https://api.minimax.chat/v1/text/chatcompletion_v2"
DEFAULT_MODEL = "MiniMax-M2"
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class MiniMaxError(Exception):
    """Base exception for MiniMax API errors."""

    pass


class MiniMaxAPIError(MiniMaxError):
    """Exception raised when MiniMax API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API error {status_code}: {message}")


class MiniMaxRateLimitError(MiniMaxError):
    """Exception raised when rate limit is exceeded."""

    pass


def _get_api_key() -> str:
    """Get the MiniMax API key from environment variable.

    Returns:
        The API key string.

    Raises:
        ValueError: If MINIMAX_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY environment variable is not set. "
            "Please set it before using the MiniMax client."
        )
    return api_key


def _make_request(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    max_tokens: Optional[int] = None,
) -> dict:
    """Make a request to the MiniMax API.

    Args:
        prompt: The prompt to send to the API.
        model: The model to use for completion.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The API response as a dictionary.

    Raises:
        MiniMaxAPIError: If the API returns an error status code.
        MiniMaxRateLimitError: If rate limit is exceeded.
        MiniMaxError: For other API-related errors.
    """
    api_key = _get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    with httpx.Client(timeout=120.0) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.post(API_ENDPOINT, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            f"Rate limit hit, retrying in {RETRY_DELAY}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        import time
                        time.sleep(RETRY_DELAY)
                        continue
                    raise MiniMaxRateLimitError(
                        "Rate limit exceeded after max retries"
                    )
                raise MiniMaxAPIError(
                    e.response.status_code,
                    e.response.text
                )
            except httpx.RequestError as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Request failed, retrying in {RETRY_DELAY}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    import time
                    time.sleep(RETRY_DELAY)
                    continue
                raise MiniMaxError(f"Request failed after {MAX_RETRIES} attempts: {e}")


def generate_code_variants(prompt: str, num_variants: int = 5) -> List[str]:
    """Generate multiple code variants based on a prompt.

    This function uses the MiniMax API to generate several different
    code implementations that satisfy the given prompt/requirements.

    Args:
        prompt: A description of the code to generate, including
            requirements and constraints.
        num_variants: The number of different variants to generate.
            Defaults to 5.

    Returns:
        A list of code variant strings.

    Raises:
        MiniMaxError: If the API request fails after retries.
        ValueError: If num_variants is less than 1.

    Example:
        >>> variants = generate_code_variants(
        ...     "Write a function to calculate fibonacci numbers",
        ...     num_variants=3
        ... )
        >>> len(variants)
        3
    """
    if num_variants < 1:
        raise ValueError("num_variants must be at least 1")

    generation_prompt = (
        f"You are an expert programmer. Generate exactly {num_variants} "
        f"different code variants that solve the following problem:\n\n"
        f"{prompt}\n\n"
        f"Return the {num_variants} variants separated by '---VARIANT---'. "
        f"Each variant should be complete, working code. "
        f"Only include the code, no explanations."
    )

    response = _make_request(
        prompt=generation_prompt,
        temperature=0.9,
        max_tokens=4000,
    )

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise MiniMaxError(f"Unexpected API response format: {e}")

    variants = content.split("---VARIANT---")
    variants = [v.strip() for v in variants if v.strip()]

    if len(variants) < num_variants:
        logger.warning(
            f"Expected {num_variants} variants but got {len(variants)}. "
            "The prompt may not have generated enough variants."
        )

    return variants[:num_variants]


def optimize_code(code: str, problem: str) -> str:
    """Optimize the given code for the specified problem.

    This function sends the code to the MiniMax API and asks it to
    optimize the code while maintaining correctness for the given problem.

    Args:
        code: The code to optimize.
        problem: A description of the problem the code should solve,
            including any constraints or requirements.

    Returns:
        The optimized code as a string.

    Raises:
        MiniMaxError: If the API request fails after retries.
        ValueError: If code or problem is empty.

    Example:
        >>> optimized = optimize_code(
        ...     "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        ...     "Calculate fibonacci numbers efficiently"
        ... )
    """
    if not code:
        raise ValueError("Code cannot be empty")
    if not problem:
        raise ValueError("Problem description cannot be empty")

    optimization_prompt = (
        "You are an expert programmer specializing in code optimization. "
        "Optimize the following code to solve the given problem more efficiently "
        "while maintaining correctness. Keep the same interface/function signatures "
        "if applicable.\n\n"
        f"Problem: {problem}\n\n"
        f"Original code:\n{code}\n\n"
        "Return ONLY the optimized code, no explanations or markdown formatting."
    )

    response = _make_request(
        prompt=optimization_prompt,
        temperature=0.7,
        max_tokens=4000,
    )

    try:
        optimized_code = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise MiniMaxError(f"Unexpected API response format: {e}")

    return optimized_code.strip()
