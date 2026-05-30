"""Prompt sampling and proposal generation for the product workbench."""

from __future__ import annotations

import os
import re
from typing import Protocol, Sequence

from .models import CandidateRecord, Proposal, TaskSpec


class ProductProposer(Protocol):
    """Generate a program proposal from a prompt and archive context."""

    def propose(
        self,
        task: TaskSpec,
        parent: CandidateRecord,
        archive: Sequence[CandidateRecord],
        prompt: str,
    ) -> Proposal:
        ...


class ProductPromptSampler:
    """Build prompts from task specs, archive state, and evaluator failures."""

    def build_prompt(
        self,
        task: TaskSpec,
        parent: CandidateRecord,
        archive: Sequence[CandidateRecord],
    ) -> str:
        failures = parent.evaluation.failures[:3]
        lines = [
            "You are AlphaEvolve-style code search controller.",
            "Task objective:",
            task.objective,
            "",
            "Function contract:",
            f"- language: {task.language}",
            f"- function: {task.function_name}",
            f"- metric: {task.metric}",
            "",
            "Constraints:",
            *[f"- {constraint}" for constraint in task.constraints],
            "",
            "Parent candidate:",
            f"- id: {parent.candidate_id}",
            f"- generation: {parent.generation}",
            f"- score: {parent.score:.6f}",
            "```python",
            parent.program,
            "```",
            "",
            "Evaluator diagnostics:",
        ]
        if failures:
            lines.extend(
                f"- {failure.case_id}: expected {failure.expected}, got {failure.actual}"
                for failure in failures
            )
        else:
            lines.append("- all sampled cases passed")

        lines.extend(["", "Archive inspirations:"])
        inspirations = [
            candidate
            for candidate in archive
            if candidate.candidate_id != parent.candidate_id
        ][:3]
        if inspirations:
            for candidate in inspirations:
                lines.append(
                    f"- {candidate.candidate_id} gen={candidate.generation} score={candidate.score:.6f}"
                )
        else:
            lines.append("- none yet")

        lines.extend(
            [
                "",
                "Evaluator cases:",
                *[
                    (
                        f"- {case.case_id}: args={repr(case.args)} "
                        f"expected={repr(case.expected)}"
                    )
                    for case in task.cases[:8]
                ],
                "",
                "Return either:",
                "- set:<complete improved python program>",
                "- replace:<old text>=><new text>",
            ]
        )
        return "\n".join(lines)


class HeuristicProductProposer:
    """Offline fallback proposer for local testing and demos."""

    _SOLUTIONS = {
        "sort_numbers": """def solve(values):
    return sorted(values)
""",
        "find_first_index": """def solve(values, target):
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1
""",
        "binary_search": """def solve(values, target):
    left = 0
    right = len(values) - 1
    while left <= right:
        middle = (left + right) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle - 1
    return -1
""",
        "two_sum_indices": """def solve(values, target):
    seen = {}
    for index, value in enumerate(values):
        needed = target - value
        if needed in seen:
            return [seen[needed], index]
        seen[value] = index
    return []
""",
        "valid_parentheses": """def solve(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack
""",
        "fibonacci": """def solve(n):
    if n <= 1:
        return n
    previous, current = 0, 1
    for _ in range(2, n + 1):
        previous, current = current, previous + current
    return current
""",
        "gcd": """def solve(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a
""",
        "is_prime": """def solve(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True
""",
        "sieve_primes": """def solve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    factor = 2
    while factor * factor <= n:
        if is_prime[factor]:
            for multiple in range(factor * factor, n + 1, factor):
                is_prime[multiple] = False
        factor += 1
    return [number for number in range(2, n + 1) if is_prime[number]]
""",
        "factorial": """def solve(n):
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
""",
        "reverse_string": """def solve(text):
    return text[::-1]
""",
        "palindrome_check": """def solve(text):
    return text == text[::-1]
""",
        "merge_intervals": """def solve(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda interval: interval[0])
    merged = [ordered[0][:]]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return merged
""",
        "max_subarray_sum": """def solve(values):
    best = values[0]
    current = values[0]
    for value in values[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best
""",
        "longest_common_subsequence": """def solve(left, right):
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for column, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]
""",
        "edit_distance": """def solve(left, right):
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[column - 1])
            else:
                current.append(1 + min(previous[column], current[-1], previous[column - 1]))
        previous = current
    return previous[-1]
""",
        "knapsack_01": """def solve(weights, values, capacity):
    best = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for current_capacity in range(capacity, weight - 1, -1):
            best[current_capacity] = max(best[current_capacity], best[current_capacity - weight] + value)
    return best[capacity]
""",
        "bfs_order": """def solve(graph, start):
    visited = {start}
    order = []
    queue = [start]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
""",
        "dijkstra_shortest_path": """def solve(graph, start):
    import heapq

    distances = {start: 0}
    queue = [(0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances
""",
        "matrix_multiply": """def solve(a, b):
    if not a or not b:
        return []
    columns = list(zip(*b))
    return [[sum(left * right for left, right in zip(row, column)) for column in columns] for row in a]
""",
    }

    def propose(
        self,
        task: TaskSpec,
        parent: CandidateRecord,
        archive: Sequence[CandidateRecord],
        prompt: str,
    ) -> Proposal:
        del archive, prompt
        solution = self._SOLUTIONS.get(task.task_id)
        if solution is None:
            return Proposal(
                text=f"append:\n# No local heuristic available for {task.task_id}.\n",
                source="heuristic",
            )
        if parent.program.strip() == solution.strip():
            return Proposal(text=f"set:{solution}", source="heuristic-noop")
        return Proposal(text=f"set:{solution}", source="heuristic")


class MiniMaxProductProposer:
    """Model-backed proposer using the existing MiniMax adapter when configured."""

    def __init__(self) -> None:
        from alphaevolve.llm_evolution.minimax_adapter import MiniMaxTextGenerator

        self._model = os.environ.get("MINIMAX_MODEL") or "MiniMax-M2.7-highspeed"
        max_tokens = int(os.environ.get("MINIMAX_MAX_TOKENS") or "6000")
        temperature = float(os.environ.get("MINIMAX_TEMPERATURE") or "0.2")
        self._generator = MiniMaxTextGenerator(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def propose(
        self,
        task: TaskSpec,
        parent: CandidateRecord,
        archive: Sequence[CandidateRecord],
        prompt: str,
    ) -> Proposal:
        del task, parent, archive
        raw_output = self._generator.generate(prompt)
        program = _extract_program(raw_output)
        if program.startswith(("set:", "replace:", "append:")):
            return Proposal(text=program, source=self._model, raw_output=raw_output)
        return Proposal(text=f"set:{program}", source=self._model, raw_output=raw_output)


def create_default_proposer() -> ProductProposer:
    """Use the real model path when configured, with a deterministic local fallback."""
    if os.environ.get("MINIMAX_API_KEY"):
        try:
            return MiniMaxProductProposer()
        except Exception:
            return HeuristicProductProposer()
    return HeuristicProductProposer()


def _extract_program(raw_output: str) -> str:
    without_thinking = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    directive_match = re.search(r"(?:^|\n)(set:|replace:|append:)", without_thinking)
    if directive_match:
        directive = directive_match.group(1)
        payload = without_thinking[directive_match.end():]
        if directive == "set:":
            return f"set:{_extract_complete_program(payload)}"
        return f"{directive}{payload.strip()}\n"

    fenced = re.search(r"```(?:python)?\s*(.*?)```", without_thinking, re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return _extract_complete_program(without_thinking)


def _extract_complete_program(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"

    code_start = _find_python_code_start(text)
    if code_start >= 0:
        return text[code_start:].strip() + "\n"
    return text.strip() + "\n"


def _find_python_code_start(text: str) -> int:
    code_start = re.search(
        r"(?m)^(?:from\s+\S+\s+import\s+|import\s+\S+|@\w+|def\s+|class\s+)",
        text,
    )
    if code_start:
        return code_start.start()
    return -1
