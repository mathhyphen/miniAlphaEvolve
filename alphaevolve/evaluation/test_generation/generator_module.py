"""Test case generator with increasing difficulty levels."""

import random
import ast
from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Difficulty(Enum):
    """Test case difficulty levels."""
    BASIC = 1        # Happy path, simple inputs
    STANDARD = 2     # Typical use cases
    EDGE = 3         # Boundary conditions
    STRESS = 4       # Large inputs
    ADVERSARIAL = 5  # Inputs designed to break


@dataclass
class TestCase:
    """A single test case."""
    inputs: Tuple[Any, ...]
    expected_output: Any
    difficulty: Difficulty = Difficulty.BASIC
    description: str = ""
    timeout: Optional[float] = None

    def to_tuple(self) -> Tuple[Tuple, Any]:
        """Convert to (input, expected) tuple."""
        return (self.inputs, self.expected_output)


class TestCaseGenerator:
    """Generate test cases of increasing difficulty."""

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self._seed = seed

    def generate(
        self,
        function_signature: str,
        difficulty: Difficulty,
        count: int = 5,
    ) -> List[TestCase]:
        """Generate test cases for a function."""
        func_name, params = self._parse_signature(function_signature)

        if difficulty == Difficulty.BASIC:
            return self._generate_basic(func_name, params, count)
        elif difficulty == Difficulty.STANDARD:
            return self._generate_standard(func_name, params, count)
        elif difficulty == Difficulty.EDGE:
            return self._generate_edge(func_name, params, count)
        elif difficulty == Difficulty.STRESS:
            return self._generate_stress(func_name, params, count)
        else:
            return self._generate_adversarial(func_name, params, count)

    def _parse_signature(self, signature: str) -> Tuple[str, List[str]]:
        """Parse function signature."""
        signature = signature.strip()
        if '(' not in signature:
            return signature, []

        func_part = signature.split('(')[0]
        params_part = signature.split('(')[1].rstrip(')')
        params = [p.strip() for p in params_part.split(',')] if params_part.strip() else []
        return func_part, params

    def _generate_basic(self, func_name: str, params: List[str], count: int) -> List[TestCase]:
        """Generate basic happy-path tests."""
        tests = []
        for i in range(count):
            if len(params) == 1:
                inputs = (random.randint(1, 10),)
            elif len(params) == 2:
                inputs = (random.randint(1, 10), random.randint(1, 10))
            else:
                inputs = tuple(random.randint(1, 10) for _ in range(len(params)))

            tests.append(TestCase(
                inputs=inputs,
                expected_output=None,
                difficulty=Difficulty.BASIC,
                description=f"Basic test {i+1}: simple positive integers",
            ))
        return tests

    def _generate_standard(self, func_name: str, params: List[str], count: int) -> List[TestCase]:
        """Generate standard use case tests."""
        tests = []
        for i in range(count):
            inputs = tuple(random.randint(-100, 100) for _ in range(len(params)))
            tests.append(TestCase(
                inputs=inputs,
                expected_output=None,
                difficulty=Difficulty.STANDARD,
                description=f"Standard test {i+1}: mixed integers",
            ))
        return tests

    def _generate_edge(self, func_name: str, params: List[str], count: int) -> List[TestCase]:
        """Generate edge case tests."""
        tests = []
        edge_values = [0, 1, -1, 10**9, -10**9]

        for i in range(count):
            inputs = tuple(random.choice(edge_values) for _ in range(len(params)))
            tests.append(TestCase(
                inputs=inputs,
                expected_output=None,
                difficulty=Difficulty.EDGE,
                description=f"Edge case test {i+1}: boundary values",
            ))

        if len(params) >= 1:
            tests.append(TestCase(
                inputs=(0,) * len(params),
                expected_output=None,
                difficulty=Difficulty.EDGE,
                description="All zeros",
            ))
        return tests

    def _generate_stress(self, func_name: str, params: List[str], count: int) -> List[TestCase]:
        """Generate stress tests with large inputs."""
        tests = []
        for i in range(count):
            if len(params) == 1:
                inputs = (list(range(10000)),)
            elif len(params) == 2:
                inputs = (list(range(1000)), list(range(1000)))
            else:
                inputs = tuple(list(range(100)) for _ in range(len(params)))

            tests.append(TestCase(
                inputs=inputs,
                expected_output=None,
                difficulty=Difficulty.STRESS,
                description=f"Stress test {i+1}: large inputs",
                timeout=5.0,
            ))
        return tests

    def _generate_adversarial(self, func_name: str, params: List[str], count: int) -> List[TestCase]:
        """Generate adversarial tests."""
        tests = []
        adversarial_inputs = [
            (None,), ([],), ("",), ((1, 2, 3),), ({"key": "val"},),
        ]

        for i in range(min(count, len(adversarial_inputs))):
            inputs = adversarial_inputs[i]
            while len(inputs) < len(params):
                inputs = inputs + (None,)

            tests.append(TestCase(
                inputs=inputs[:len(params)],
                expected_output=None,
                difficulty=Difficulty.ADVERSARIAL,
                description=f"Adversarial test {i+1}: unusual input",
                timeout=2.0,
            ))
        return tests
