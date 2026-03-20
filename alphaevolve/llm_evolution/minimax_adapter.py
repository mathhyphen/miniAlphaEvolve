"""MiniMax LLM adapter for the AlphaEvolve-style evolution framework.

This module connects the MiniMax LLM to the TextGenerator protocol
defined in llm_evolution/__init__.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import TextGenerator


class MiniMaxTextGenerator:
    """TextGenerator implementation using MiniMax LLM API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-M2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize MiniMax text generator.

        Args:
            api_key: MiniMax API key. If None, reads from MINIMAX_API_KEY env.
            model: Model name.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
        """
        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY", "")
            if not api_key:
                raise ValueError("MINIMAX_API_KEY environment variable not set")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Generate text from prompt using MiniMax LLM.

        Args:
            prompt: Input prompt for the LLM.

        Returns:
            Generated text response.
        """
        from alphaevolve.llm.minimax_client import generate_code_variants

        try:
            # Use the existing generate_code_variants function
            # It returns a list, we take the first variant
            variants = generate_code_variants(
                prompt=prompt,
                num_variants=1,
            )
            if variants:
                return variants[0]
            return ""
        except Exception as e:
            raise RuntimeError(f"MiniMax generation failed: {e}") from e


class MiniMaxDiffProposer:
    """A DiffProposer that uses MiniMax LLM for code generation.

    This combines:
    - PromptSampler: builds prompts from parent and archive
    - TextGenerator: generates proposals via MiniMax LLM
    """

    def __init__(
        self,
        text_generator: TextGenerator,
        prompt_sampler=None,
        diff_style: str = "set",
    ):
        """Initialize proposer.

        Args:
            text_generator: TextGenerator instance (e.g., MiniMaxTextGenerator)
            prompt_sampler: PromptSampler for building prompts
            diff_style: Preferred diff style ("set", "replace", "append")
        """
        self.text_generator = text_generator
        self.prompt_sampler = prompt_sampler
        self.diff_style = diff_style

    def propose(self, parent, archive) -> str:
        """Generate a code proposal.

        Args:
            parent: ProgramCandidate to improve upon
            archive: ProgramDatabase with inspirations

        Returns:
            A diff proposal string
        """
        if self.prompt_sampler:
            from . import PromptSampler
            inspirations = archive.inspirations_for(parent, limit=2)
            prompt = self.prompt_sampler.build_prompt(parent, inspirations, archive)
        else:
            prompt = f"""Improve the following code for better performance:

Parent (score={parent.score}):
```
{parent.program}
```

Return the complete improved code using 'set:<full program>' format."""

        result = self.text_generator.generate(prompt)

        # Format as diff
        if self.diff_style == "set" and not result.startswith("set:"):
            return f"set:\n{result}"
        return result


# Factory function for convenience
def create_minimax_proposer(
    api_key: str | None = None,
    temperature: float = 0.7,
    prompt_sampler=None,
) -> "MiniMaxDiffProposer":
    """Create a MiniMax-backed diff proposer.

    Args:
        api_key: MiniMax API key. If None, reads from MINIMAX_API_KEY env.
        temperature: LLM temperature for generation.
        prompt_sampler: Optional PromptSampler instance.

    Returns:
        Configured MiniMaxDiffProposer instance.
    """
    text_gen = MiniMaxTextGenerator(
        api_key=api_key,
        temperature=temperature,
    )

    return MiniMaxDiffProposer(
        text_generator=text_gen,
        prompt_sampler=prompt_sampler,
    )


__all__ = [
    "MiniMaxTextGenerator",
    "MiniMaxDiffProposer",
    "create_minimax_proposer",
]
