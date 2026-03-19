"""Mutation engine for AlphaEvolve using LLM-based code modification."""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from alphaevolve.core.data_structures import MutationResult
from alphaevolve.llm.ensemble import LLMEnsemble
from alphaevolve.llm.client import LLMResponse

logger = logging.getLogger(__name__)


# Default system prompt for code evolution
DEFAULT_SYSTEM_PROMPT = """You are an expert Python programmer. CRITICAL RULES:
1. ALWAYS output COMPLETE, RUNNABLE Python code
2. Every variable used must be defined in the output
3. Every function call must exist
4. Code must pass: exec(code) without errors
5. Return ONLY code in ```python``` blocks, NO explanations"""


# Prompt templates for different mutation strategies
MUTATION_PROMPTS = {
    "improve_performance": """You are an expert Python programmer.

CRITICAL RULES:
1. You MUST keep ALL functions, variables, and logic from the original code
2. You can ONLY add small optimizations (like replacing bubble sort with sorted())
3. You CANNOT remove or simplify any existing functionality
4. The output code MUST produce the SAME result as the original

Original code:
{code}

{feedback}

Your task: Make ONE small optimization (e.g., replace bubble sort with sorted()).

Output the COMPLETE code with only the small change:
```python
""",

    "fix_bugs": """You are an expert Python programmer.

CRITICAL RULES:
1. You MUST keep ALL functions, variables, and logic from the original code
2. You can ONLY fix the specific bug mentioned in feedback
3. You CANNOT change any other part of the code
4. The output code MUST produce the SAME result as the original

Original code:
{code}

{feedback}

Your task: Fix ONLY the bug mentioned.

Output the COMPLETE code with only the bug fix:
```python
""",

    "simplify": """You are an expert Python programmer.

CRITICAL RULES:
1. You MUST keep ALL functions, variables, and logic from the original code
2. You can ONLY make small readability improvements
3. You CANNOT change any algorithm or logic
4. The output code MUST produce the SAME result as the original

Original code:
{code}

{feedback}

Your task: Make only small readability improvements.

Output the COMPLETE code:
```python
""",

    "add_features": """You are an expert Python programmer.

CRITICAL RULES:
1. You MUST keep ALL existing functions and logic from the original code
2. You can ONLY add the requested feature
3. You CANNOT modify existing functionality
4. The output code MUST produce the SAME result as the original

Original code:
{code}

{feedback}

Your task: Add ONLY the requested feature.

Output the COMPLETE code:
```python
""",

    "general": """You are an expert Python programmer.

CRITICAL RULES:
1. You MUST output the COMPLETE original code with MINIMAL changes
2. You can make at most ONE small improvement
3. Do NOT rewrite the entire code - just make ONE small change
4. The output MUST run and produce correct results

Original code:
{code}

{feedback}

Your task: Make ONE small targeted improvement.

Output the COMPLETE code with minimal changes:
```python
"""
}


class MutationStrategy(ABC):
    """Abstract base for code mutation strategies."""

    @abstractmethod
    def can_apply(self, code: str) -> bool:
        """Check if this strategy can be applied to the code."""
        pass

    @abstractmethod
    def apply(self, code: str) -> str:
        """Apply the mutation to the code."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PathCompressionStrategy(MutationStrategy):
    """Add path compression to find function."""

    def can_apply(self, code: str) -> bool:
        return "def find(parent, i):" in code and "parent[i] = find" not in code

    def apply(self, code: str) -> str:
        # Pattern 1: Simple recursive return without path compression
        old_pattern = r'(def find\(parent, i\):.*?if parent\[i\] == i:.*?return i)(.*?)(return find\(parent, i\))'
        new_code = r'\1\2parent[i] = find(parent, i)\n    return parent[i]'

        result = re.sub(old_pattern, new_code, code, flags=re.DOTALL)

        # Pattern 2: Alternative form with != check
        if result == code:
            old_pattern2 = r'(def find\(parent, i\):.*?if parent\[i\] != i:.*?)(parent\[i\] = find\(parent, parent\[i\]\))'
            if 'parent[i] = find' not in code:
                simple_find_pattern = r'(def find\(parent, i\):)\n(.*?)(return find\(parent, i\))'
                result = re.sub(simple_find_pattern,
                               r'\1\n\2parent[i] = find(parent, i)\n    return parent[i]',
                               code, flags=re.DOTALL)

        return result if result != code else code


class BubbleSortToSortedStrategy(MutationStrategy):
    """Replace bubble sort with built-in sorted()."""

    def can_apply(self, code: str) -> bool:
        return "sorted_edges = list(edges)" in code or "n = len(sorted_edges)" in code

    def apply(self, code: str) -> str:
        # Match bubble sort pattern
        bubble_sort_pattern = r'sorted_edges = list\(edges\)\n\s*n = len\(sorted_edges\)\n\s*for i in range\(n\):.*?for j in range\(0, n - i - 1\):.*?if sorted_edges\[j\]\[2\] > sorted_edges\[j \+ 1\]\[2\]:.*?sorted_edges\[j\], sorted_edges\[j \+ 1\] = sorted_edges\[j \+ 1\], sorted_edges\[j\]'

        if re.search(bubble_sort_pattern, code, re.DOTALL):
            return re.sub(bubble_sort_pattern,
                         'sorted_edges = sorted(edges, key=lambda x: x[2])',
                         code, flags=re.DOTALL)

        # Try simpler pattern
        simple_pattern = r'sorted_edges = list\(edges\)\n\s*n = len\(sorted_edges\)'
        if re.search(simple_pattern, code):
            code = re.sub(simple_pattern, 'sorted_edges = sorted(edges, key=lambda x: x[2])', code)
            # Remove the bubble sort loops
            code = re.sub(r'\n\s*for i in range\(n\):.*?sorted_edges\[j\], sorted_edges\[j \+ 1\] = sorted_edges\[j \+ 1\], sorted_edges\[j\]',
                         '', code, flags=re.DOTALL)

        return code


class EarlyTerminationStrategy(MutationStrategy):
    """Add early termination when MST is complete."""

    def can_apply(self, code: str) -> bool:
        return "for edge in sorted_edges:" in code and "edges_count += 1" in code

    def apply(self, code: str) -> str:
        # Find the edge processing loop and add early termination
        pattern = r'(for edge in sorted_edges:.*?if union\(parent, rank, u, v\):.*?edges_count \+= 1)'
        replacement = r'\1\n            if edges_count == vertices - 1:\n                break'

        result = re.sub(pattern, replacement, code, flags=re.DOTALL)
        return result


class RecursionOptimizationStrategy(MutationStrategy):
    """Convert recursive find to iterative version."""

    def can_apply(self, code: str) -> bool:
        recursive_pattern = r'def find\(parent, i\):.*?if parent\[i\] == i:.*?return i.*?return find\(parent, i\)'
        return bool(re.search(recursive_pattern, code, re.DOTALL)) and 'while parent[i] != i:' not in code

    def apply(self, code: str) -> str:
        iterative_find = '''def find(parent, i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path compression
            i = parent[i]
        return i'''

        recursive_pattern = r'def find\(parent, i\):.*?if parent\[i\] == i:.*?return i.*?return find\(parent, i\)'

        if re.search(recursive_pattern, code, re.DOTALL) and 'while parent[i] != i:' not in code:
            return re.sub(recursive_pattern, iterative_find, code, flags=re.DOTALL)

        return code


class CommentsStrategy(MutationStrategy):
    """Add optimization comments."""

    def can_apply(self, code: str) -> bool:
        return True  # Can always add comments

    def apply(self, code: str) -> str:
        lines = code.split('\n')
        if lines:
            lines[0] = "# Optimized implementation\n" + lines[0]
        return '\n'.join(lines)


class ConditionalRestructureStrategy(MutationStrategy):
    """Restructure conditional logic in union function."""

    def can_apply(self, code: str) -> bool:
        return 'def union' in code and 'rank' in code

    def apply(self, code: str) -> str:
        # Optimize union by rank comparison
        old_union = r'if root_x != root_y:\n\s*if rank\[root_x\] < rank\[root_y\]:\n\s*parent\[root_x\] = root_y\n\s*else:\n\s*parent\[root_y\] = root_x\n\s*if rank\[root_x\] == rank\[root_y\]:\n\s*rank\[root_x\] \+= 1'

        # This is already a good pattern, just clean up formatting
        if re.search(old_union, code, re.DOTALL):
            return code  # Already optimal

        # Try to add rank comparison optimization
        if 'def union' in code and 'rank' in code:
            if 'rank[root_x] < rank[root_y]' not in code:
                code = code.replace(
                    'if root_x != root_y:',
                    'if root_x != root_y:\n        if rank[root_x] < rank[root_y]:\n            parent[root_x] = root_y\n        else:\n            parent[root_y] = root_x\n            if rank[root_x] == rank[root_y]:\n                rank[root_x] += 1'
                )

        return code


@dataclass
class MutationEngineConfig:
    """Configuration for mutation engine.

    Args:
        temperature: LLM temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds
        max_retries: Maximum retry attempts
    """
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 3


class MutationEngine:
    """Engine for generating code mutations using LLM."""

    def __init__(
        self,
        llm_ensemble: Optional[LLMEnsemble] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize mutation engine.

        Args:
            llm_ensemble: LLM ensemble for code generation
            system_prompt: Optional system prompt for LLM
        """
        self._llm_ensemble = llm_ensemble
        self.mutation_types = list(MUTATION_PROMPTS.keys())
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._validation_test_cases: list = None

    def set_validation_test_cases(self, test_cases: List[Tuple[Any, Any]]) -> None:
        """Set test cases for validation.

        Args:
            test_cases: List of (test_input, expected) tuples
        """
        self._validation_test_cases = test_cases

    def _default_system_prompt(self) -> str:
        """Return default system prompt for code evolution."""
        return DEFAULT_SYSTEM_PROMPT

    def mutate(
        self,
        code: str,
        mutation_type: Optional[str] = None,
        feedback: str = "",
        parent_ids: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> MutationResult:
        """Generate a mutation of the given code.

        Args:
            code: Original code
            mutation_type: Type of mutation (or random if None)
            feedback: Feedback from evaluator
            parent_ids: IDs of parent individuals
            max_retries: Maximum number of retries on validation failure

        Returns:
            MutationResult with mutated code
        """
        import random

        # Select mutation type
        if mutation_type is None:
            mutation_type = random.choice(self.mutation_types)

        if mutation_type not in MUTATION_PROMPTS:
            return MutationResult(
                code=code,
                success=False,
                explanation=f"Unknown mutation type: {mutation_type}",
                mutation_type=mutation_type,
                parent_ids=parent_ids or []
            )

        last_error = None

        for attempt in range(max_retries):
            # Build feedback with retry info
            retry_feedback = feedback
            if attempt > 0 and last_error:
                retry_feedback = f"{feedback}\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease fix this error and provide COMPLETE working code."

            # Generate prompt
            prompt = MUTATION_PROMPTS[mutation_type].format(
                code=code,
                feedback=retry_feedback if retry_feedback else "No specific feedback provided. Make sure ALL variables are defined and the code is complete."
            )

            # Call LLM
            try:
                mutated_code = self._call_llm(prompt)

                # Validate syntax
                syntax_valid = self._validate_code(mutated_code)

                # Validate execution
                if syntax_valid:
                    execution_result = self._validate_code_execution(
                        mutated_code,
                        validation_test_cases=self._validation_test_cases
                    )
                    if isinstance(execution_result, tuple):
                        execution_valid, error_msg = execution_result
                        last_error = error_msg
                    else:
                        execution_valid = execution_result
                        last_error = "exec() failed with error"
                else:
                    execution_valid = False
                    last_error = "Syntax validation failed"

                success = syntax_valid and execution_valid

                if success:
                    return MutationResult(
                        code=mutated_code,
                        success=True,
                        explanation=f"Applied {mutation_type} mutation",
                        confidence=0.8,
                        mutation_type=mutation_type,
                        parent_ids=parent_ids or []
                    )

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"Mutation attempt {attempt + 1} failed: {last_error}")

        # All retries exhausted
        return MutationResult(
            code=code,
            success=False,
            explanation=f"Failed after {max_retries} attempts. Last error: {last_error}",
            mutation_type=mutation_type,
            parent_ids=parent_ids or []
        )

    def mutate_batch(
        self,
        code: str,
        count: int = 3,
        feedback: str = "",
        parent_ids: Optional[List[str]] = None
    ) -> List[MutationResult]:
        """Generate multiple mutations.

        Args:
            code: Original code
            count: Number of mutations to generate
            feedback: Feedback from evaluator
            parent_ids: IDs of parent individuals

        Returns:
            List of mutation results
        """
        results = []
        for mutation_type in self.mutation_types[:count]:
            result = self.mutate(code, mutation_type, feedback, parent_ids)
            results.append(result)
        return results

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate mutation.

        Args:
            prompt: The mutation prompt

        Returns:
            Generated code
        """
        # Use LLM ensemble if available
        if self._llm_ensemble is not None:
            try:
                response = self._llm_ensemble.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                )
                return self._extract_code_from_response(response.content)
            except Exception as e:
                logger.warning(f"LLM ensemble failed: {e}, using fallback")
                return self._fallback_mutate(prompt)
        else:
            # Use fallback when no LLM configured
            return self._fallback_mutate(prompt)

    def set_llm_ensemble(self, ensemble: LLMEnsemble) -> None:
        """Configure LLM ensemble for mutations.

        Args:
            ensemble: LLM ensemble to use
        """
        self._llm_ensemble = ensemble
        logger.info("LLM ensemble configured")

    def _extract_code_from_response(self, content: str) -> str:
        """Extract code from LLM response.

        Args:
            content: LLM response content

        Returns:
            Extracted code or original content
        """
        # Try to extract code from markdown block
        code_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try without language specifier
        code_match = re.search(r'```\n(.*?)```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Return content as-is if no code block found
        return content.strip()

    def _fallback_mutate(self, prompt: str) -> str:
        """Fallback mutation for testing/demo purposes.

        This applies simple heuristics to modify code without LLM.
        In production, this would be replaced with actual LLM calls.

        Args:
            prompt: The mutation prompt (contains code in markdown block)

        Returns:
            Modified code
        """
        import random

        # Extract code from prompt (between triple backticks)
        code_match = re.search(r'```python\n(.*?)```', prompt, re.DOTALL)
        if not code_match:
            # Try without language specifier
            code_match = re.search(r'```\n(.*?)```', prompt, re.DOTALL)

        if not code_match:
            return "# Error: Could not extract code"

        code = code_match.group(1).strip()

        # Define fallback strategies
        fallback_strategies: List[MutationStrategy] = [
            PathCompressionStrategy(),
            BubbleSortToSortedStrategy(),
            EarlyTerminationStrategy(),
            RecursionOptimizationStrategy(),
            CommentsStrategy(),
            ConditionalRestructureStrategy(),
        ]

        # Apply 1-3 random mutations
        num_mutations = random.randint(1, 3)
        applicable = [s for s in fallback_strategies if s.can_apply(code)]
        selected_strategies = random.sample(applicable, min(num_mutations, len(applicable)))

        result = code
        for strategy in selected_strategies:
            new_result = strategy.apply(result)
            # Only apply if it actually changes the code
            if new_result != result:
                result = new_result
                logger.debug(f"Applied strategy: {strategy.name}")

        return result

    def _validate_code(self, code: str) -> bool:
        """Validate that generated code is syntactically correct.

        Args:
            code: Code to validate

        Returns:
            True if valid Python
        """
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _validate_code_execution(
        self,
        code: str,
        expected_functions: list = None,
        validation_test_cases: list = None
    ) -> tuple:
        """Validate that generated code can be executed and produces correct output.

        Args:
            code: Code to validate
            expected_functions: List of function names that should exist in the code.
            validation_test_cases: Optional test cases (test_input, expected) to verify correctness.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to execute the code in an isolated namespace
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            # Check if expected functions exist
            if expected_functions:
                has_function = any(
                    name in namespace
                    for name in expected_functions
                )
                if not has_function:
                    return (False, "Generated code missing expected function")

            # If we have test cases, run them
            if validation_test_cases and "steiner_tree" in namespace:
                func = namespace["steiner_tree"]
                for test_input, expected in validation_test_cases:
                    vertices, edges, terminals = test_input
                    try:
                        result = func(vertices, edges, terminals)
                        if abs(result - expected) >= 0.001:
                            return (False, f"Test failed: expected {expected}, got {result}")
                    except Exception as e:
                        return (False, f"Runtime error: {type(e).__name__}: {str(e)}")

            return (True, None)
        except SyntaxError as e:
            return (False, f"Syntax error: {str(e)}")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return (False, error_msg)
