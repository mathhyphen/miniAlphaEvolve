"""Patch generator for RL-based code modification.

Generates code patches (edits) based on policy network outputs,
focusing on graph theory problems like Steiner Tree.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import torch

from .policy_network import PolicyNetwork, PolicyConfig, compute_advantages

logger = logging.getLogger(__name__)


class PatchType(Enum):
    """Types of code patches that can be generated."""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    WRAP = "wrap"
    SWAP = "swap"
    PARAM_CHANGE = "param_change"


@dataclass(frozen=True)
class Patch:
    """Represents a code patch/action.

    Args:
        patch_type: Type of patch
        target_line: Target line number for patch
        content: New content (for insert/replace/wrap)
        end_line: End line for delete/wrap operations
        description: Human-readable description
        confidence: Policy confidence in this patch (0.0-1.0)
    """
    patch_type: PatchType
    target_line: int
    content: str = ""
    end_line: Optional[int] = None
    description: str = ""
    confidence: float = 0.5

    def apply(self, code: str) -> str:
        """Apply this patch to the given code.

        Args:
            code: Original code string

        Returns:
            Modified code string

        Raises:
            ValueError: If patch cannot be applied
        """
        lines = code.splitlines()
        num_lines = len(lines)

        if self.target_line < 0 or self.target_line >= num_lines:
            raise ValueError(f"Target line {self.target_line} out of range [0, {num_lines})")

        if self.patch_type == PatchType.INSERT:
            lines.insert(self.target_line, self.content)
        elif self.patch_type == PatchType.DELETE:
            end = self.end_line if self.end_line is not None else self.target_line + 1
            if end > num_lines:
                end = num_lines
            del lines[self.target_line:end]
        elif self.patch_type == PatchType.REPLACE:
            lines[self.target_line] = self.content
        elif self.patch_type == PatchType.WRAP:
            end = self.end_line if self.end_line is not None else self.target_line + 1
            if end > num_lines:
                end = num_lines
            lines[self.target_line] = f"{self.content} {lines[self.target_line]}"
            if end <= num_lines:
                lines[end - 1] = f"{lines[end - 1]} {self._get_closing()}"
        elif self.patch_type == PatchType.SWAP:
            if self.end_line is None or self.end_line >= num_lines:
                raise ValueError(f"Swap target {self.end_line} out of range")
            lines[self.target_line], lines[self.end_line] = lines[self.end_line], lines[self.target_line]
        elif self.patch_type == PatchType.PARAM_CHANGE:
            lines[self.target_line] = self._apply_param_change(lines[self.target_line])

        return "\n".join(lines)

    def _get_closing(self) -> str:
        """Get closing bracket/marker for wrap operation."""
        if "{" in self.content:
            return "}"
        if "(" in self.content:
            return ")"
        if "[" in self.content:
            return "]"
        return ""

    def _apply_param_change(self, line: str) -> str:
        """Apply parameter change to a line."""
        if "def " in line or "class " in line:
            return self.content
        match = re.search(r'(\w+)\s*=\s*', line)
        if match:
            var_name = match.group(1)
            return re.sub(rf'{var_name}\s*=\s*[^,\)]+', f'{var_name}={self.content}', line)
        return line


@dataclass
class ExecutionResult:
    """Result of executing code on a test case.

    Args:
        passed: Whether execution passed
        fitness: Fitness score achieved
        execution_time: Time taken to execute
        error: Error message if failed
        output: Output from execution
    """
    passed: bool
    fitness: float
    execution_time: float = 0.0
    error: str = ""
    output: str = ""


@dataclass
class CodeState:
    """Represents the current state of code for RL.

    Args:
        code: Current code string
        execution_history: List of execution results
        fitness_history: Historical fitness scores
        graph_features: Features extracted from graph problem
    """
    code: str
    execution_history: List[ExecutionResult] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    graph_features: Optional[List[float]] = None

    def get_reward(self) -> float:
        """Compute reward from current state.

        Returns:
            Reward value based on execution results
        """
        if not self.execution_history:
            return -0.1

        total_fitness = sum(r.fitness for r in self.execution_history)
        avg_fitness = total_fitness / len(self.execution_history)

        all_passed = all(r.passed for r in self.execution_history)
        if all_passed:
            return 1.0 + avg_fitness

        return avg_fitness


class PatchGenerator:
    """Generates code patches using RL policy.

    The patch generator uses a trained policy network to generate
    code modifications targeting graph theory problems like Steiner Tree.
    """

    def __init__(
        self,
        policy: Optional[PolicyNetwork] = None,
        config: Optional[PolicyConfig] = None,
        max_patch_attempts: int = 10,
    ) -> None:
        """Initialize patch generator.

        Args:
            policy: Pre-trained policy network (created if None)
            config: Policy configuration (used if policy is None)
            max_patch_attempts: Maximum patch generation attempts
        """
        if policy is None:
            if config is None:
                config = PolicyConfig()
            self.policy = PolicyNetwork(config)
        else:
            self.policy = policy

        self.max_patch_attempts = max_patch_attempts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def tokenize_code(self, code: str, max_len: int = 512) -> torch.Tensor:
        """Tokenize code into integer tokens.

        Args:
            code: Code string to tokenize
            max_len: Maximum token sequence length

        Returns:
            Token tensor of shape (1, max_len)
        """
        tokens = []
        for char in code[:max_len]:
            tokens.append(ord(char) % 10000)
        while len(tokens) < max_len:
            tokens.append(0)
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def extract_graph_features(self, code: str, problem_type: str = "steiner") -> torch.Tensor:
        """Extract graph-theoretic features from code.

        Args:
            code: Code to analyze
            problem_type: Type of graph problem

        Returns:
            Feature tensor of shape (1, 64)
        """
        features = []

        line_count = len(code.splitlines())
        features.append(min(line_count / 100.0, 1.0))

        features.append(1.0 if "for " in code or "while " in code else 0.0)
        features.append(1.0 if "if " in code else 0.0)
        features.append(1.0 if "def " in code else 0.0)

        features.append(code.count("return") / 10.0)
        features.append(code.count("for ") / 10.0)
        features.append(code.count("while ") / 10.0)

        edge_pattern = re.findall(r'\([^)]*,\s*[^)]*,\s*\d+\)', code)
        features.append(min(len(edge_pattern) / 20.0, 1.0))

        features.append(1.0 if "heapq" in code or "PriorityQueue" in code else 0.0)
        features.append(1.0 if "union" in code.lower() or "disjoint" in code.lower() else 0.0)

        features.append(1.0 if "recursive" in code.lower() or "recursion" in code.lower() else 0.0)
        features.append(1.0 if "memo" in code.lower() or "cache" in code.lower() else 0.0)

        terminal_pattern = re.findall(r'terminal|sink|source|endpoint', code.lower())
        features.append(min(len(terminal_pattern) / 5.0, 1.0))

        features.append(code.count("[") / 20.0)
        features.append(code.count("]") / 20.0)
        features.append(code.count("(") / 20.0)
        features.append(code.count(")") / 20.0)

        weight_pattern = re.findall(r'(?:weight|cost|dist)[\s]*[=:]?\s*\d+', code.lower())
        features.append(min(len(weight_pattern) / 10.0, 1.0))

        features.append(1.0 if "sorted" in code or "sort" in code else 0.0)
        features.append(1.0 if "min(" in code or "max(" in code else 0.0)

        while len(features) < 64:
            features.append(0.0)

        return torch.tensor([features[:64]], dtype=torch.float32, device=self.device)

    def encode_execution_history(self, history: List[ExecutionResult]) -> torch.Tensor:
        """Encode execution history into feature tensor.

        Args:
            history: List of execution results

        Returns:
            History tensor of shape (1, history_len, 32)
        """
        max_history = 10
        history_features = []

        for result in history[-max_history:]:
            features = [
                1.0 if result.passed else 0.0,
                min(result.fitness / 10.0, 1.0),
                min(result.execution_time / 60.0, 1.0),
                1.0 if result.error else 0.0,
            ]
            features.extend([0.0] * 28)
            history_features.append(features)

        while len(history_features) < max_history:
            history_features.append([0.0] * 32)

        return torch.tensor([history_features], dtype=torch.float32, device=self.device)

    def action_to_patch(
        self,
        action: int,
        code: str,
        confidence: float = 0.5,
    ) -> Patch:
        """Convert policy action index to Patch object.

        Args:
            action: Action index from policy
            code: Current code context
            confidence: Policy confidence score

        Returns:
            Patch object representing the action
        """
        lines = code.splitlines()
        num_lines = max(len(lines), 1)
        target_line = action % num_lines

        action_type = (action // num_lines) % len(PatchType)

        patch_type = PatchType(action_type)

        if patch_type == PatchType.INSERT:
            content = self._sample_code_snippet("insert")
        elif patch_type == PatchType.REPLACE:
            content = self._sample_code_snippet("replace")
        elif patch_type == PatchType.WRAP:
            content = self._sample_code_snippet("wrap")
        elif patch_type == PatchType.PARAM_CHANGE:
            content = str(random.choice([1, 2, 5, 10, 0.1, 0.5]))
        else:
            content = ""

        return Patch(
            patch_type=patch_type,
            target_line=target_line,
            content=content,
            end_line=(target_line + 1) if patch_type in [PatchType.DELETE, PatchType.WRAP] else None,
            description=f"{patch_type.value} at line {target_line}",
            confidence=confidence,
        )

    def _sample_code_snippet(self, snippet_type: str) -> str:
        """Sample a code snippet for insertion.

        Args:
            snippet_type: Type of snippet to sample

        Returns:
            Code snippet string
        """
        snippets = {
            "insert": [
                "if dist[u] + w < dist[v]:",
                "heapq.heappush(pq, (dist[v], v))",
                "parent[v] = u",
                "if v not in visited:",
                "for neighbor in graph[u]:",
                "if weight < min_edge:",
                "total += edge_weight",
                "if is_terminal(v):",
            ],
            "replace": [
                "dist[u] + w",
                "float('inf')",
                "heapq.heappop(pq)",
                "visited.add(v)",
                "min(dist.items(), key=lambda x: x[1])",
                "sum(w for _, _, w in edges)",
            ],
            "wrap": [
                "try:",
                "if True:",
                "while True:",
                "for _ in range(1):",
            ],
        }

        return random.choice(snippets.get(snippet_type, ["pass"]))

    def generate_patch(
        self,
        state: CodeState,
        deterministic: bool = False,
    ) -> Patch:
        """Generate a single patch from current state.

        Args:
            state: Current code state
            deterministic: If True, take most likely action

        Returns:
            Generated patch
        """
        code_tokens = self.tokenize_code(state.code)
        code_mask = (code_tokens != 0).float()

        history_tensor = None
        if state.execution_history:
            history_tensor = self.encode_execution_history(state.execution_history)

        graph_features = None
        if state.graph_features:
            graph_features = torch.tensor(
                [state.graph_features[:64]],
                dtype=torch.float32,
                device=self.device
            )
        else:
            graph_features = self.extract_graph_features(state.code)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(
                code_tokens,
                code_mask,
                history_tensor,
                graph_features,
                deterministic=deterministic,
            )

        confidence = float(torch.exp(log_prob).cpu().item())

        return self.action_to_patch(
            action.item(),
            state.code,
            confidence=confidence,
        )

    def generate_patches(
        self,
        state: CodeState,
        num_patches: int = 5,
        deterministic: bool = False,
    ) -> List[Patch]:
        """Generate multiple patches from current state.

        Args:
            state: Current code state
            num_patches: Number of patches to generate
            deterministic: If True, take most likely actions

        Returns:
            List of generated patches
        """
        patches = []
        for _ in range(num_patches):
            try:
                patch = self.generate_patch(state, deterministic=deterministic)
                patches.append(patch)
            except Exception as e:
                logger.warning(f"Patch generation failed: {e}")
                continue

        return patches

    def update_policy(
        self,
        states: List[CodeState],
        actions: List[int],
        rewards: List[float],
        old_log_probs: List[float],
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
    ) -> Dict[str, float]:
        """Update policy using collected experiences.

        Args:
            states: List of code states
            actions: List of actions taken
            rewards: List of rewards received
            old_log_probs: List of old log probabilities
            gamma: Discount factor
            lam: GAE lambda
            clip_epsilon: PPO clipping epsilon

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()

        code_tokens_list = []
        code_mask_list = []
        history_list = []
        graph_features_list = []

        for state in states:
            code_tokens = self.tokenize_code(state.code)
            code_mask = (code_tokens != 0).float()
            code_tokens_list.append(code_tokens)
            code_mask_list.append(code_mask)

            if state.execution_history:
                history_list.append(self.encode_execution_history(state.execution_history))
            else:
                history_list.append(torch.zeros(1, 10, 32, device=self.device))

            if state.graph_features:
                graph_features_list.append(torch.tensor(
                    [state.graph_features[:64]],
                    dtype=torch.float32,
                    device=self.device
                ))
            else:
                graph_features_list.append(self.extract_graph_features(state.code))

        code_tokens_batch = torch.cat(code_tokens_list, dim=0)
        code_mask_batch = torch.cat(code_mask_list, dim=0)
        history_batch = torch.cat(history_list, dim=0)
        graph_batch = torch.cat(graph_features_list, dim=0)

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)

        log_probs, entropy, values = self.policy.evaluate_actions(
            code_tokens_batch,
            actions_tensor,
            code_mask_batch,
            history_batch,
            graph_batch,
        )

        advantages, returns = compute_advantages(rewards, values.cpu().tolist(), gamma, lam)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns_tensor)

        entropy_loss = -entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
        }
