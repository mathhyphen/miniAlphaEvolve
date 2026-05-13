"""Formal verification module for algorithm contracts and assertions.

This module provides lightweight formal verification capabilities
for checking algorithm properties and contracts.
"""

import dataclasses
import functools
import logging
import typing as T
from dataclasses import field

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Contract:
    """A contract specifying preconditions and postconditions."""
    name: str
    description: str
    preconditions: T.List[T.Callable[[T.Any, ...], bool]] = field(default_factory=list)
    postconditions: T.List[T.Callable[[T.Any, T.Any], bool]] = field(default_factory=list)
    invariants: T.List[T.Callable[[T.Any], bool]] = field(default_factory=list)


@dataclasses.dataclass
class VerificationResult:
    """Result of a verification check."""
    passed: bool
    contract_name: str
    check_type: str  # "precondition", "postcondition", "invariant", "assertion"
    message: str
    details: T.Optional[T.Dict[str, T.Any]] = None


@dataclasses.dataclass
class VerificationSummary:
    """Summary of all verification results."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: T.List[VerificationResult]
    execution_time_ms: float


class ContractError(Exception):
    """Raised when a contract is violated."""
    pass


class PreconditionError(ContractError):
    """Raised when a precondition is not met."""
    pass


class PostconditionError(ContractError):
    """Raised when a postcondition is not met."""
    pass


class InvariantError(ContractError):
    """Raised when an invariant is not met."""
    pass


def precondition(func: T.Callable[..., bool]) -> T.Callable[..., T.Any]:
    """Decorator to mark a function as a precondition checker.

    Args:
        func: A function that takes the same arguments as the decorated
              function and returns True if precondition is satisfied.

    Returns:
        Decorated precondition function.

    Example:
        @precondition
        def graph_is_connected(graph):
            return is_connected(graph)
    """
    @functools.wraps(func)
    def wrapper(*args: T.Any, **kwargs: T.Any) -> bool:
        result = func(*args, **kwargs)
        if not result:
            logger.warning(f"Precondition {func.__name__} failed")
        return result
    return wrapper


def postcondition(func: T.Callable[..., bool]) -> T.Callable[..., T.Any]:
    """Decorator to mark a function as a postcondition checker.

    Args:
        func: A function that takes the result and original arguments
              and returns True if postcondition is satisfied.

    Returns:
        Decorated postcondition function.

    Example:
        @postcondition
        def solution_is_valid(result, *args):
            return verify_steiner_tree(result, *args)
    """
    @functools.wraps(func)
    def wrapper(result: T.Any, *args: T.Any, **kwargs: T.Any) -> bool:
        check_result = func(result, *args, **kwargs)
        if not check_result:
            logger.warning(f"Postcondition {func.__name__} failed")
        return check_result
    return wrapper


def invariant(func: T.Callable[[T.Any], bool]) -> T.Callable[[T.Any], T.Any]:
    """Decorator to mark a function as an invariant checker.

    Args:
        func: A function that takes an object and returns True if
              the invariant is satisfied.

    Returns:
        Decorated invariant function.

    Example:
        @invariant
        def tree_property(obj):
            return is_tree(obj) and has_expected_nodes(obj)
    """
    @functools.wraps(func)
    def wrapper(obj: T.Any) -> bool:
        result = func(obj)
        if not result:
            logger.warning(f"Invariant {func.__name__} failed")
        return result
    return wrapper


class ContractVerifier:
    """Verify contracts and algorithm properties.

    This class provides utilities for checking contracts during
    algorithm execution and collecting verification results.

    Example:
        verifier = ContractVerifier()

        @verifier.register_contract
        def steiner_tree_contract(graph, terminals):
            return Contract(
                name="steiner_tree",
                description="Steiner tree solution validation",
                preconditions=[
                    lambda g, t: len(t) >= 2,
                    lambda g, t: all(node_exists(g, n) for n in t),
                ],
                postconditions=[
                    lambda r, g, t: is_connected(r),
                    lambda r, g, t: all(is_terminal_covered(r, n) for n in t),
                ],
            )

        result = verifier.verify_contract(
            steiner_tree_contract(graph, terminals),
            lambda: solve_steiner(graph, terminals)
        )
    """

    def __init__(self, fail_fast: bool = False) -> None:
        """Initialize the contract verifier.

        Args:
            fail_fast: If True, raise exception on first failure.
        """
        self._contracts: T.Dict[str, Contract] = {}
        self._fail_fast = fail_fast

    def register_contract(self, contract: Contract) -> None:
        """Register a contract for verification.

        Args:
            contract: The contract to register.
        """
        self._contracts[contract.name] = contract
        logger.debug(f"Registered contract: {contract.name}")

    def add_precondition(
        self, contract_name: str, precondition: T.Callable[..., bool]
    ) -> None:
        """Add a precondition to an existing contract.

        Args:
            contract_name: Name of the contract to extend.
            precondition: Function to check as precondition.
        """
        if contract_name not in self._contracts:
            raise ValueError(f"Contract '{contract_name}' not found")
        self._contracts[contract_name].preconditions.append(precondition)

    def add_postcondition(
        self, contract_name: str, postcondition: T.Callable[..., bool]
    ) -> None:
        """Add a postcondition to an existing contract.

        Args:
            contract_name: Name of the contract to extend.
            postcondition: Function to check as postcondition.
        """
        if contract_name not in self._contracts:
            raise ValueError(f"Contract '{contract_name}' not found")
        self._contracts[contract_name].postconditions.append(postcondition)

    def verify_preconditions(
        self, contract: Contract, args: T.Tuple[T.Any, ...], kwargs: T.Dict[str, T.Any]
    ) -> T.List[VerificationResult]:
        """Verify all preconditions of a contract.

        Args:
            contract: The contract to verify.
            args: Arguments to pass to precondition checkers.
            kwargs: Keyword arguments to pass to precondition checkers.

        Returns:
            List of verification results.
        """
        results: T.List[VerificationResult] = []

        for precond in contract.preconditions:
            try:
                passed = precond(*args, **kwargs)
                results.append(VerificationResult(
                    passed=passed,
                    contract_name=contract.name,
                    check_type="precondition",
                    message=f"Precondition {precond.__name__ if hasattr(precond, '__name__') else 'anonymous'}",
                    details={"passed": passed},
                ))
                if not passed and self._fail_fast:
                    raise PreconditionError(f"Precondition failed: {precond}")
            except Exception as e:
                results.append(VerificationResult(
                    passed=False,
                    contract_name=contract.name,
                    check_type="precondition",
                    message=f"Precondition check raised exception",
                    details={"error": str(e)},
                ))

        return results

    def verify_postconditions(
        self,
        contract: Contract,
        result: T.Any,
        args: T.Tuple[T.Any, ...],
        kwargs: T.Dict[str, T.Any],
    ) -> T.List[VerificationResult]:
        """Verify all postconditions of a contract.

        Args:
            contract: The contract to verify.
            result: The result to verify.
            args: Original arguments passed to the function.
            kwargs: Original keyword arguments passed to the function.

        Returns:
            List of verification results.
        """
        results: T.List[VerificationResult] = []

        for postcond in contract.postconditions:
            try:
                passed = postcond(result, *args, **kwargs)
                results.append(VerificationResult(
                    passed=passed,
                    contract_name=contract.name,
                    check_type="postcondition",
                    message=f"Postcondition {postcond.__name__ if hasattr(postcond, '__name__') else 'anonymous'}",
                    details={"passed": passed},
                ))
                if not passed and self._fail_fast:
                    raise PostconditionError(f"Postcondition failed: {postcond}")
            except Exception as e:
                results.append(VerificationResult(
                    passed=False,
                    contract_name=contract.name,
                    check_type="postcondition",
                    message=f"Postcondition check raised exception",
                    details={"error": str(e)},
                ))

        return results

    def verify_invariants(
        self, contract: Contract, state: T.Any
    ) -> T.List[VerificationResult]:
        """Verify all invariants of a contract.

        Args:
            contract: The contract to verify.
            state: The state object to check invariants against.

        Returns:
            List of verification results.
        """
        results: T.List[VerificationResult] = []

        for inv in contract.invariants:
            try:
                passed = inv(state)
                results.append(VerificationResult(
                    passed=passed,
                    contract_name=contract.name,
                    check_type="invariant",
                    message=f"Invariant {inv.__name__ if hasattr(inv, '__name__') else 'anonymous'}",
                    details={"passed": passed},
                ))
                if not passed and self._fail_fast:
                    raise InvariantError(f"Invariant failed: {inv}")
            except Exception as e:
                results.append(VerificationResult(
                    passed=False,
                    contract_name=contract.name,
                    check_type="invariant",
                    message=f"Invariant check raised exception",
                    details={"error": str(e)},
                ))

        return results

    def verify_contract(
        self,
        contract: Contract,
        func: T.Callable[..., T.Any],
        args: T.Tuple[T.Any, ...] = (),
        kwargs: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> VerificationSummary:
        """Verify a contract by executing a function and checking all conditions.

        Args:
            contract: The contract to verify.
            func: The function to execute and verify.
            args: Arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.

        Returns:
            VerificationSummary with all results.
        """
        import time
        start_time = time.perf_counter()

        if kwargs is None:
            kwargs = {}

        all_results: T.List[VerificationResult] = []

        # Check preconditions
        pre_results = self.verify_preconditions(contract, args, kwargs)
        all_results.extend(pre_results)

        failed_preconditions = [r for r in pre_results if not r.passed]
        if failed_preconditions:
            # Skip execution if preconditions fail
            if self._fail_fast:
                raise PreconditionError("Preconditions not satisfied")
            return VerificationSummary(
                total_checks=len(all_results),
                passed_checks=sum(1 for r in all_results if r.passed),
                failed_checks=len(failed_preconditions),
                results=all_results,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Execute function
        result: T.Any = None
        execution_error: T.Optional[str] = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            execution_error = f"{type(e).__name__}: {e}"
            if self._fail_fast:
                raise

        # Check postconditions if execution succeeded
        if execution_error is None and result is not None:
            post_results = self.verify_postconditions(contract, result, args, kwargs)
            all_results.extend(post_results)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return VerificationSummary(
            total_checks=len(all_results),
            passed_checks=sum(1 for r in all_results if r.passed),
            failed_checks=sum(1 for r in all_results if not r.passed),
            results=all_results,
            execution_time_ms=execution_time_ms,
        )

    def verify_assertion(
        self,
        condition: bool,
        message: str,
        details: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> VerificationResult:
        """Verify a simple assertion.

        Args:
            condition: The condition to check.
            message: Description of the assertion.
            details: Optional additional details.

        Returns:
            VerificationResult for the assertion.
        """
        result = VerificationResult(
            passed=condition,
            contract_name="assertion",
            check_type="assertion",
            message=message,
            details=details,
        )

        if not condition and self._fail_fast:
            raise ContractError(f"Assertion failed: {message}")

        return result


# Standard contract templates for graph algorithms

def create_steiner_tree_contract() -> Contract:
    """Create a standard contract for Steiner tree algorithms.

    Returns:
        Contract with standard Steiner tree verification conditions.
    """
    return Contract(
        name="steiner_tree_standard",
        description="Standard contract for Steiner tree algorithms",
        preconditions=[],
        postconditions=[],
        invariants=[],
    )


def verify_steiner_tree_properties(
    solution: T.Dict[str, T.Any],
    graph: T.Dict[str, T.Any],
    terminals: T.List[int],
) -> T.Tuple[bool, str]:
    """Verify basic properties of a Steiner tree solution.

    Args:
        solution: Dictionary containing:
            - edges: List of (u, v, weight) tuples
            - total_cost: Total weight of the tree
        graph: Graph representation with 'nodes' and 'edges'.
        terminals: List of terminal node IDs.

    Returns:
        Tuple of (is_valid, error_message).
    """
    edges = solution.get("edges", [])
    total_cost = solution.get("total_cost", 0.0)

    # Check all terminals are covered
    covered_nodes = set()
    for u, v, _ in edges:
        covered_nodes.add(u)
        covered_nodes.add(v)

    uncovered = [t for t in terminals if t not in covered_nodes]
    if uncovered:
        return False, f"Uncovered terminals: {uncovered}"

    # Check graph is connected (simple check using edge count)
    num_connected_components = _count_connected_components(edges, graph.get("nodes", []))
    if num_connected_components > 1:
        return False, f"Graph is not connected, has {num_connected_components} components"

    # Check total cost matches sum of edge weights
    computed_cost = sum(w for _, _, w in edges)
    if abs(computed_cost - total_cost) > 1e-9:
        return False, f"Cost mismatch: claimed {total_cost}, computed {computed_cost}"

    # Check no cycles in a simple way (n-1 edges for n nodes)
    num_nodes_in_tree = len(covered_nodes)
    if len(edges) != num_nodes_in_tree - 1:
        return False, f"Invalid tree: {len(edges)} edges for {num_nodes_in_tree} nodes (expected {num_nodes_in_tree - 1})"

    return True, "Valid Steiner tree solution"


def _count_connected_components(
    edges: T.List[T.Tuple[int, int, float]], nodes: T.List[int]
) -> int:
    """Count connected components using Union-Find."""
    if not nodes:
        return 0

    parent = {n: n for n in nodes}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for u, v, _ in edges:
        if u in parent and v in parent:
            union(u, v)

    components = set(find(n) for n in nodes)
    return len(components)
