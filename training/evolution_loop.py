"""Evolution Loop for AlphaEvolve RL training - PPO-based code patch evolution."""

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from alphaevolve.rl.policy_network import PolicyConfig, PolicyNetwork
from alphaevolve.rl.patch_generator import (
    CodeState,
    ExecutionResult,
    Patch,
    PatchGenerator,
    PatchSample,
)
from alphaevolve.sandbox.executor import ExecutionConfig, ExecutionResult as ExecResult, SandboxExecutor
from alphaevolve.sandbox.metrics import MetricsCollector, PerformanceTracker
from alphaevolve.archive.pwa import PWArchiveConfig, PopulationWideArchive

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration for evolution loop."""
    num_iterations: int = 1000
    episodes_per_iteration: int = 4
    patches_per_episode: int = 8
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    baseline_latency: float = 100.0
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    episode_id: str
    iteration: int
    initial_code: str
    final_code: str
    patches: List[Patch]
    rewards: List[float]
    total_reward: float
    latency_ms: float
    improvement_ratio: float
    trajectory: List["TrajectoryStep"]


@dataclass(frozen=True)
class TrajectoryStep:
    """A single PPO rollout step captured from the live trajectory."""

    state_code: str
    execution_history: List[ExecutionResult]
    fitness_history: List[float]
    graph_features: Optional[List[float]]
    action: int
    old_log_prob: float
    value: float
    patch: Patch
    reward: float


class EvolutionLoop:
    """Main PPO-based evolution loop for AlphaEvolve RL training.

    Performs: Patch generation -> Sandbox execution -> Latency measurement
    -> Reward computation -> PWA Archive update -> Checkpoint saving.
    """

    def __init__(self, problem: Any, config: Optional[EvolutionConfig] = None) -> None:
        """Initialize evolution loop."""
        self.problem = problem
        self.config = config or EvolutionConfig()

        # Policy network
        policy_cfg = PolicyConfig()
        self.policy = PolicyNetwork(policy_cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Patch generator
        self.patch_gen = PatchGenerator(policy=self.policy, config=policy_cfg,
                                        max_patch_attempts=self.config.patches_per_episode)

        # Sandbox executor
        self.executor = SandboxExecutor()
        self.exec_config = ExecutionConfig(timeout_seconds=30.0, memory_limit_mb=512.0)

        # Metrics
        self.metrics = MetricsCollector(window_size=10000)
        self.tracker = PerformanceTracker(self.metrics)

        # PWA Archive
        archive_cfg = PWArchiveConfig(baseline_latency=self.config.baseline_latency,
                                      max_snapshots_per_cell=10, max_total_snapshots=1000)
        self.archive = PopulationWideArchive(archive_cfg)

        # Training state
        self.current_iteration = 0
        self.episode_count = 0
        self.best_improvement = 0.0

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._set_seed(42)
        logger.info(f"EvolutionLoop initialized on {self.device}")

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_initial_state(self, problem: Any) -> CodeState:
        """Create initial code state for a problem."""
        code = getattr(problem, "initial_code", "") or self._get_default_code(problem)
        graph_features = getattr(problem, "graph_features", None)
        return CodeState(code=code, execution_history=[], fitness_history=[], graph_features=graph_features)

    def _get_default_code(self, problem: Any) -> str:
        """Get default initial code."""
        return """def solve(n, edges, terminals):
    # Steiner Tree Solver
    return []
"""

    def _generate_patches(self, state: CodeState, num_patches: int) -> List[PatchSample]:
        """Generate patches using policy network."""
        samples: List[PatchSample] = []
        for _ in range(num_patches):
            try:
                sample = self.patch_gen.generate_patch(state, deterministic=False)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Patch generation failed: {e}")
        return samples

    def _execute_patch(self, code: str, patch: Patch, problem: Any) -> Tuple[ExecResult, float]:
        """Execute patched code in sandbox."""
        try:
            patched_code = patch.apply(code)
        except ValueError as e:
            return ExecResult(success=False, output=None, error=str(e), execution_time=0.0, peak_memory_mb=0.0, exit_code=-1), 0.0

        test_case = getattr(problem, "get_test_case", lambda: {})()
        args = test_case.get("args", (getattr(problem, "num_nodes", 10), [], []))
        func_name = getattr(problem, "function_name", "solve")

        result = self.executor.execute(code=patched_code, function=func_name, args=args, config=self.exec_config)
        latency_ms = result.execution_time * 1000.0
        self.metrics.record_latency("patch_execution", latency_ms)
        return result, latency_ms

    def _compute_reward(self, exec_result: ExecResult, latency_ms: float, baseline: float) -> float:
        """Compute reward from execution result."""
        if not exec_result.success:
            return -1.0
        if latency_ms < baseline:
            return (1.0 - latency_ms / baseline) + 0.5
        return -0.1

    def _run_episode(self, problem: Any, iteration: int) -> EpisodeResult:
        """Run a single episode."""
        episode_id = f"ep_{self.episode_count}_{int(time.time())}"
        state = self._create_initial_state(problem)
        patches, rewards = [], []
        trajectory: List[TrajectoryStep] = []

        for _ in range(self.config.patches_per_episode):
            step_samples = self._generate_patches(state, num_patches=1)
            if not step_samples:
                continue

            sample = step_samples[0]
            patch = sample.patch
            patches.append(patch)

            state_snapshot = CodeState(
                code=state.code,
                execution_history=copy.deepcopy(state.execution_history),
                fitness_history=list(state.fitness_history),
                graph_features=list(state.graph_features) if state.graph_features else None,
            )
            exec_result, latency_ms = self._execute_patch(state.code, patch, problem)

            reward = self._compute_reward(exec_result, latency_ms, self.config.baseline_latency)
            rewards.append(reward)
            trajectory.append(
                TrajectoryStep(
                    state_code=state_snapshot.code,
                    execution_history=state_snapshot.execution_history,
                    fitness_history=state_snapshot.fitness_history,
                    graph_features=state_snapshot.graph_features,
                    action=sample.action,
                    old_log_prob=sample.log_prob,
                    value=sample.value,
                    patch=patch,
                    reward=reward,
                )
            )

            if exec_result.success:
                try:
                    state.code = patch.apply(state.code)
                except ValueError:
                    pass
                state.execution_history.append(ExecutionResult(passed=True, fitness=latency_ms, execution_time=latency_ms / 1000.0))
            else:
                state.execution_history.append(ExecutionResult(passed=False, fitness=float('inf'), execution_time=0.0, error=exec_result.error or "Unknown"))
            state.fitness_history.append(latency_ms)

        total_reward = sum(rewards)
        final_latency = state.fitness_history[-1] if state.fitness_history else self.config.baseline_latency
        improvement_ratio = 1.0 - (final_latency / self.config.baseline_latency)

        return EpisodeResult(
            episode_id=episode_id, iteration=iteration, initial_code=self._create_initial_state(problem).code,
            final_code=state.code, patches=patches, rewards=rewards, total_reward=total_reward,
            latency_ms=final_latency, improvement_ratio=improvement_ratio, trajectory=trajectory
        )

    def _update_policy(self, episode_results: List[EpisodeResult]) -> Dict[str, float]:
        """Update policy using PPO."""
        states, actions, rewards_list, old_log_probs, values = [], [], [], [], []
        for result in episode_results:
            for step in result.trajectory:
                states.append(
                    CodeState(
                        code=step.state_code,
                        execution_history=copy.deepcopy(step.execution_history),
                        fitness_history=list(step.fitness_history),
                        graph_features=list(step.graph_features) if step.graph_features else None,
                    )
                )
                actions.append(step.action)
                rewards_list.append(step.reward)
                old_log_probs.append(step.old_log_prob)
                values.append(step.value)

        if not states:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        return self.patch_gen.update_policy(states=states, actions=actions, rewards=rewards_list,
                                           old_log_probs=old_log_probs, rollout_values=values, gamma=self.config.gamma,
                                           lam=self.config.lam, clip_epsilon=self.config.clip_epsilon)

    def _update_archive(self, episode_results: List[EpisodeResult], iteration: int) -> None:
        """Update PWA archive with episode results."""
        for result in episode_results:
            if result.final_code:
                self.archive.add(code=result.final_code, latency=result.latency_ms, generation=iteration,
                                 metadata={"episode_id": result.episode_id, "num_patches": len(result.patches)})
                if result.improvement_ratio > self.best_improvement:
                    self.best_improvement = result.improvement_ratio

    def _save_checkpoint(self, iteration: int, metrics: Optional[Dict[str, float]] = None) -> str:
        """Save training checkpoint."""
        ckpt_path = self.checkpoint_dir / f"checkpoint_{iteration}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "iteration": iteration, "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.patch_gen.optimizer.state_dict(),
            "best_improvement": self.best_improvement, "episode_count": self.episode_count,
            "config": {"num_iterations": self.config.num_iterations, "gamma": self.config.gamma, "lam": self.config.lam}
        }, ckpt_path / "policy.pt")

        self.archive.save(str(ckpt_path / "archive.json"))
        if metrics:
            with open(ckpt_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        logger.info(f"Checkpoint saved to {ckpt_path}")
        return str(ckpt_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path / "policy.pt")
        self.policy.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            self.patch_gen.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_iteration = state["iteration"]
        self.best_improvement = state.get("best_improvement", 0.0)
        self.episode_count = state.get("episode_count", 0)
        self.archive = PopulationWideArchive.load(str(checkpoint_path / "archive.json"))
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def train(self, num_iterations: Optional[int] = None, resume_from: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run evolution training loop."""
        if resume_from:
            self._load_checkpoint(resume_from)

        num_iters = num_iterations or self.config.num_iterations
        results: List[Dict[str, Any]] = []

        logger.info(f"Starting training for {num_iters} iterations")

        for iteration in range(self.current_iteration, num_iters):
            iter_start = time.time()
            self.current_iteration = iteration
            episode_results: List[EpisodeResult] = []

            for ep_idx in range(self.config.episodes_per_iteration):
                ep_result = self._run_episode(self.problem, iteration)
                episode_results.append(ep_result)
                self.episode_count += 1

            policy_metrics = self._update_policy(episode_results)
            self._update_archive(episode_results, iteration)

            archive_stats = self.archive.get_stats()
            best_imp = max(r.improvement_ratio for r in episode_results)

            results.append({
                "iteration": iteration, "episode_count": len(episode_results),
                "policy_loss": policy_metrics.get("policy_loss", 0.0),
                "value_loss": policy_metrics.get("value_loss", 0.0),
                "best_improvement": best_imp, "total_snapshots": archive_stats.total_snapshots
            })

            iter_time = time.time() - iter_start
            logger.info(f"Iter {iteration}/{num_iters} ({iter_time:.1f}s) | "
                        f"Best: {self.best_improvement:.4f} | Archive: {archive_stats.total_snapshots} | "
                        f"Policy loss: {policy_metrics.get('policy_loss', 0.0):.4f}")

            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(iteration, policy_metrics)

        logger.info("Training completed!")
        return results

    def get_best_solution(self):
        """Get best solution from archive."""
        return self.archive.get_best_overall()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        archive_stats = self.archive.get_stats()
        return {
            "current_iteration": self.current_iteration, "total_episodes": self.episode_count,
            "best_improvement": self.best_improvement,
            "archive": {"total_snapshots": archive_stats.total_snapshots, "occupied_cells": archive_stats.occupied_cells,
                        "fill_rate": archive_stats.fill_rate, "strategies": archive_stats.strategies_used}
        }
