"""Code generation module for AlphaEvolve.

This module provides code variation generation, optimization, and algorithm
suggestion capabilities using the MiniMax LLM API.

Example prompting strategies supported:
    - "Generate 5 different implementations of 4x4 matrix multiplication"
    - "Optimize this code while keeping it correct"
    - "Find a faster algorithm for matrix multiplication"
"""

import logging
import re
from typing import List, Optional

from alphaevolve.llm.minimax_client import (
    MiniMaxError,
    generate_code_variants,
    optimize_code as optimize_code_api,
)

logger = logging.getLogger(__name__)

CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


class CodeGeneratorError(Exception):
    """Base exception for code generator errors."""

    pass


class CodeParseError(CodeGeneratorError):
    """Exception raised when code parsing fails."""

    pass


def _extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from LLM response text.

    Args:
        text: The raw text response from the LLM.

    Returns:
        A list of extracted code strings, cleaned and stripped.
    """
    matches = CODE_BLOCK_PATTERN.findall(text)
    if not matches:
        lines = text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.startswith(("def ", "class ", "import ", "from ", "async ", "async def")):
                code_lines.append(line)
        if code_lines:
            return ["\n".join(code_lines).strip()]
        return [text.strip()]
    return [match.strip() for match in matches if match.strip()]


def _build_variants_prompt(problem_description: str, base_code: Optional[str], num_variants: int) -> str:
    """Build prompt for generating code variants.

    Args:
        problem_description: Description of the problem to solve.
        base_code: Optional existing code implementation to vary.
        num_variants: Number of variants to generate.

    Returns:
        Formatted prompt string for the LLM.
    """
    prompt_parts = [
        f"You are an expert programmer. Generate exactly {num_variants} different",
        f"code variants that solve the following problem:\n\n{problem_description}",
    ]

    if base_code:
        prompt_parts.append(f"\n\nHere is an existing implementation to vary from:\n```\n{base_code}\n```")

    prompt_parts.append(
        f"\n\nReturn the {num_variants} variants separated by '---VARIANT---'. "
        "Each variant should be complete, working code. "
        "Only include the code, no explanations."
    )

    return "".join(prompt_parts)


def _build_optimize_prompt(code: str, problem_type: str) -> str:
    """Build prompt for optimizing code.

    Args:
        code: The code to optimize.
        problem_type: The type/kind of problem being solved.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        "You are an expert programmer specializing in code optimization. "
        "Optimize the following code to solve the given problem more efficiently "
        "while maintaining correctness. Keep the same interface/function signatures "
        "if applicable.\n\n"
        f"Problem type: {problem_type}\n\n"
        f"Code to optimize:\n```\n{code}\n```\n\n"
        "Return ONLY the optimized code, no explanations or markdown formatting."
    )


def _build_algorithm_prompt(problem: str) -> str:
    """Build prompt for suggesting algorithms.

    Args:
        problem: The problem description.

    Returns:
        Formatted prompt string for the LLM.
    """
    return (
        "You are an expert algorithm designer. Analyze the following problem "
        "and suggest the most efficient algorithm to solve it. "
        "Provide both the algorithm name/approach and implementation.\n\n"
        f"Problem:\n{problem}\n\n"
        "Return the algorithm suggestion as code, focusing on the core algorithm. "
        "Include a brief comment describing the algorithm approach."
    )


class CodeVariationGenerator:
    """Generate code variants for optimization using LLM.

    This class provides methods to generate diverse code implementations,
    optimize existing code, and suggest algorithms for problems.

    Attributes:
        temperature: Sampling temperature for LLM generation (higher = more diverse).
        max_tokens: Maximum tokens to generate per request.
    """

    def __init__(self, temperature: float = 0.8, max_tokens: int = 4000) -> None:
        """Initialize the code variation generator.

        Args:
            temperature: Sampling temperature for generation. Defaults to 0.8.
            max_tokens: Maximum tokens per generation. Defaults to 4000.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_variants(
        self,
        problem_description: str,
        base_code: str,
        num_variants: int,
    ) -> List[str]:
        """Generate diverse code variants for a problem.

        Args:
            problem_description: Description of the problem to solve.
            base_code: Existing base implementation to create variants from.
            num_variants: Number of different variants to generate.

        Returns:
            List of code variant strings.

        Raises:
            CodeGeneratorError: If generation fails after retries.
            ValueError: If inputs are invalid.
        """
        if not problem_description:
            raise ValueError("Problem description cannot be empty")
        if num_variants < 1:
            raise ValueError("num_variants must be at least 1")

        prompt = _build_variants_prompt(problem_description, base_code, num_variants)

        try:
            raw_variants = generate_code_variants(prompt, num_variants)
        except MiniMaxError as e:
            raise CodeGeneratorError(f"Failed to generate variants: {e}") from e

        variants = []
        for variant in raw_variants:
            extracted = _extract_code_blocks(variant)
            variants.extend(extracted)

        while len(variants) < num_variants:
            logger.warning(
                f"Requested {num_variants} variants but only got {len(variants)}. "
                "Duplicating last variant to meet quota."
            )
            variants.append(variants[-1] if variants else "")

        return variants[:num_variants]

    def optimize_code(self, code: str, problem_type: str) -> str:
        """Optimize existing code for a given problem type.

        Args:
            code: The code to optimize.
            problem_type: Type of problem being solved (e.g., "matrix multiplication").

        Returns:
            The optimized code as a string.

        Raises:
            CodeGeneratorError: If optimization fails.
            ValueError: If inputs are invalid.
        """
        if not code:
            raise ValueError("Code cannot be empty")
        if not problem_type:
            raise ValueError("Problem type cannot be empty")

        prompt = _build_optimize_prompt(code, problem_type)

        try:
            response = optimize_code_api(code, problem_type)
        except MiniMaxError as e:
            raise CodeGeneratorError(f"Failed to optimize code: {e}") from e

        extracted = _extract_code_blocks(response)
        return extracted[0] if extracted else response.strip()

    def suggest_algorithm(self, problem: str) -> str:
        """Suggest an efficient algorithm for a problem.

        Args:
            problem: Description of the problem.

        Returns:
            The suggested algorithm implementation as code.

        Raises:
            CodeGeneratorError: If suggestion generation fails.
            ValueError: If problem is empty.
        """
        if not problem:
            raise ValueError("Problem description cannot be empty")

        prompt = _build_algorithm_prompt(problem)

        try:
            raw_response = generate_code_variants(prompt, num_variants=1)
        except MiniMaxError as e:
            raise CodeGeneratorError(f"Failed to suggest algorithm: {e}") from e

        if not raw_response:
            raise CodeGeneratorError("Empty response from LLM")

        extracted = _extract_code_blocks(raw_response[0])
        return extracted[0] if extracted else raw_response[0].strip()
