"""Feature dimensions for MAP-Elites archive.

Feature dimensions are used to characterize behavior of programs
for quality-diversity search.
"""

import abc
import ast
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass


class FeatureDimension(abc.ABC):
    """Abstract base class for feature dimensions.

    Feature dimensions map programs to discrete cells in the archive.
    Each dimension defines a behavioral characteristic that can be measured.
    """

    @abc.abstractmethod
    def extract(self, code: str) -> float:
        """Extract feature value from code.

        Args:
            code: Source code string

        Returns:
            Feature value (will be discretized)
        """
        pass

    @abc.abstractmethod
    def get_bins(self) -> int:
        """Get number of bins for this feature.

        Returns:
            Number of discrete bins
        """
        pass

    @abc.abstractmethod
    def get_range(self) -> Tuple[float, float]:
        """Get expected range of feature values.

        Returns:
            Tuple of (min_value, max_value)
        """
        pass

    def discretize(self, value: float) -> int:
        """Convert continuous value to discrete bin.

        Args:
            value: Continuous feature value

        Returns:
            Bin index (0 to get_bins()-1)
        """
        min_val, max_val = self.get_range()
        n_bins = self.get_bins()

        # Clamp value to range
        value = max(min_val, min(max_val, value))

        # Convert to bin index
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        bin_index = int(normalized * (n_bins - 1))

        return max(0, min(n_bins - 1, bin_index))


class CodeComplexityFeature(FeatureDimension):
    """Feature based on cyclomatic complexity.

    Measures the number of linearly independent paths through code.
    Higher complexity = more bins.
    """

    def __init__(self, n_bins: int = 10, max_complexity: int = 50) -> None:
        """Initialize complexity feature.

        Args:
            n_bins: Number of discrete bins
            max_complexity: Maximum complexity value
        """
        self._n_bins = n_bins
        self._max_complexity = max_complexity

    def extract(self, code: str) -> float:
        """Extract cyclomatic complexity from code."""
        try:
            tree = ast.parse(code)
            return self._calculate_complexity(tree)
        except SyntaxError:
            return float('inf')  # Invalid code has max complexity

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity.

        Complexity = 1 + number of decision points
        Decision points: if, elif, for, while, and, or, except, with
        """
        complexity = 1

        for node in ast.walk(tree):
            # Decision points
            if isinstance(node, (
                ast.If,
                ast.For,
                ast.While,
                ast.ExceptHandler,
                ast.With,
                ast.AsyncFor,
                ast.AsyncWith,
            )):
                complexity += 1
            # Boolean operators
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            # Comprehensions with conditions
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for generator in node.generators:
                    complexity += len(generator.ifs)

        return complexity

    def get_bins(self) -> int:
        return self._n_bins

    def get_range(self) -> Tuple[float, float]:
        return (0.0, float(self._max_complexity))


class PerformanceFeature(FeatureDimension):
    """Feature based on execution performance.

    Measures execution time and assigns to performance bins.
    Faster code = lower bin indices.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_time: float = 0.0,
        max_time: float = 10.0,
    ) -> None:
        """Initialize performance feature.

        Args:
            n_bins: Number of discrete bins
            min_time: Minimum execution time (seconds)
            max_time: Maximum execution time (seconds)
        """
        self._n_bins = n_bins
        self._min_time = min_time
        self._max_time = max_time

    def extract(self, code: str) -> float:
        """Extract performance metric from code.

        Note: This is a placeholder. Actual performance measurement
        requires executing the code with a benchmark function.

        Returns:
            Execution time in seconds (or estimated complexity)
        """
        # Fallback to complexity-based estimate
        try:
            tree = ast.parse(code)
            complexity = CodeComplexityFeature().extract(code)
            # Estimate: complexity * 0.1 seconds
            return complexity * 0.1
        except SyntaxError:
            return self._max_time

    def get_bins(self) -> int:
        return self._n_bins

    def get_range(self) -> Tuple[float, float]:
        return (self._min_time, self._max_time)


class CodeSizeFeature(FeatureDimension):
    """Feature based on code size.

    Measures lines of code (LOC).
    """

    def __init__(self, n_bins: int = 10, max_lines: int = 200) -> None:
        """Initialize code size feature.

        Args:
            n_bins: Number of discrete bins
            max_lines: Maximum lines of code
        """
        self._n_bins = n_bins
        self._max_lines = max_lines

    def extract(self, code: str) -> float:
        """Extract lines of code."""
        lines = code.strip().split('\n')
        # Count non-empty, non-comment lines
        loc = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                loc += 1
        return float(loc)

    def get_bins(self) -> int:
        return self._n_bins

    def get_range(self) -> Tuple[float, float]:
        return (0.0, float(self._max_lines))


class BehavioralFeature(FeatureDimension):
    """Feature based on output behavior signature.

    Creates a hash-based signature of program output.
    Programs with similar outputs map to nearby cells.
    """

    def __init__(self, n_bins: int = 16) -> None:
        """Initialize behavioral feature.

        Args:
            n_bins: Number of discrete bins (should be power of 2 for hashing)
        """
        self._n_bins = n_bins

    def extract(self, code: str) -> float:
        """Extract behavioral signature.

        Note: This requires executing the code with test inputs.
        Returns a hash-based feature value.
        """
        import hashlib

        try:
            # Execute and hash output
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            # Find callable and run with sample input
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    func = obj
                    break

            if func:
                try:
                    output = str(func(1, 2))  # Sample execution
                except:
                    output = "error"
            else:
                output = "no_function"

            # Hash the output
            hash_bytes = hashlib.sha256(output.encode()).digest()
            # Use first byte as feature value
            return float(hash_bytes[0])

        except Exception:
            return 0.0

    def get_bins(self) -> int:
        return self._n_bins

    def get_range(self) -> Tuple[float, float]:
        return (0.0, 255.0)  # Byte value range
