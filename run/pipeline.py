"""AlphaEvolve Redesign - CLI Pipeline Entry Point.

Usage:
    python -m run.pipeline --problem steiner_tree --iterations 1000
    python -m run.pipeline --problem mst --checkpoint checkpoint.pt
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from alphaevolve.archive import PWArchiveConfig, PopulationWideArchive
from alphaevolve.rl.patch_generator import CodeState, ExecutionResult, PatchGenerator
from alphaevolve.rl.policy_network import PolicyConfig, PolicyNetwork
from alphaevolve.sandbox.executor import ExecutionConfig, SandboxExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    description: str
    baseline_code: str
    baseline_function: str
    baseline_latency: float
    test_cases: List[Any]


PROBLEMS: Dict[str, ProblemConfig] = {}


def register_problem(name: str, desc: str, code: str, func: str,
                     latency: float, tests: Optional[List[Any]] = None) -> None:
    PROBLEMS[name] = ProblemConfig(name=name, description=desc, baseline_code=code,
                                   baseline_function=func, baseline_latency=latency, test_cases=tests or [])


register_problem("steiner_tree", "Euclidean Steiner Tree",
    '''def steiner_tree(terminals):
    import math
    if len(terminals) < 2:
        return 0.0
    remaining = set(range(1, len(terminals)))
    in_tree = {0}
    total = 0.0
    while remaining:
        best_index = None
        best_distance = float("inf")
        for source in in_tree:
            sx, sy = terminals[source]
            for target in remaining:
                tx, ty = terminals[target]
                distance = math.hypot(sx - tx, sy - ty)
                if distance < best_distance:
                    best_distance = distance
                    best_index = target
        in_tree.add(best_index)
        remaining.remove(best_index)
        total += best_distance
    return total''',
    "steiner_tree", 100.0, [[(0,0),(1,1),(2,0)], [(0,0),(1,0),(0.5,0.866)]])

register_problem("mst", "Minimum Spanning Tree",
    '''def mst(edges):
    if not edges:
        return 0.0
    max_node = max(max(u, v) for u, v, _ in edges)
    parent = list(range(max_node + 1))
    rank = [0] * (max_node + 1)
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1
        return True
    total = 0.0
    for left, right, weight in sorted(edges, key=lambda edge: edge[2]):
        if union(left, right):
            total += weight
    return total''',
    "mst", 50.0, [[(0,1,1),(1,2,2),(0,2,3)], [(0,1,5),(1,2,3),(2,3,1)]])


class EvolutionLoop:
    def __init__(self, problem: ProblemConfig, iterations: int = 1000,
                 output_dir: Optional[str] = None, checkpoint: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        self.problem = problem
        self.iterations = iterations
        self.output_dir = Path(output_dir) if output_dir else self._mk_output_dir()
        self.seed = seed
        if seed: random.seed(seed); torch.manual_seed(seed)

        self.policy = PolicyNetwork(PolicyConfig())
        self.patch_gen = PatchGenerator(policy=self.policy)
        self.executor = SandboxExecutor()
        self.archive = PopulationWideArchive(PWArchiveConfig(baseline_latency=problem.baseline_latency, max_total_snapshots=500))

        self.code = problem.baseline_code
        self.state = CodeState(code=self.code)
        self.gen = 0
        self.best = 0.0
        self.t0 = time.time()
        self.history: List[Dict] = []
        if checkpoint: self._load(checkpoint)

    def _mk_output_dir(self) -> Path:
        p = Path(f"outputs/{self.problem.name}_{datetime.now():%Y%m%d_%H%M%S}")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _eval(self, code: str) -> ExecutionResult:
        r = self.executor.execute(
            code=code,
            function=self.problem.baseline_function,
            args=((self.problem.test_cases[0] if self.problem.test_cases else []),),
            config=ExecutionConfig(timeout_seconds=30.0),
        )
        fit = 1.0 / (1.0 + r.output) if r.success and r.output else 0.0
        return ExecutionResult(passed=r.success, fitness=fit, execution_time=r.execution_time, error=r.error or "")

    def _iterate(self) -> None:
        self.gen += 1
        patches = self.patch_gen.generate_patches(self.state, num_patches=5)
        best_code, best_fit = self.code, self.best

        for p in patches:
            try:
                nc = p.apply(self.code)
                res = self._eval(nc)
                if res.fitness > best_fit:
                    best_fit, best_code = res.fitness, nc
            except Exception:
                continue

        if best_code != self.code:
            self.code = best_code
            self.state = CodeState(code=self.code, execution_history=self.state.execution_history +
                                   [ExecutionResult(passed=best_fit > 0, fitness=best_fit)])
            if best_fit > self.best:
                self.best = best_fit
                logger.info(f"Gen {self.gen}: new best = {self.best:.4f}")
            self.archive.add(code=self.code, latency=1.0/best_fit if best_fit > 0 else float('inf'), generation=self.gen)
        self.history.append({"gen": self.gen, "best": self.best, "t": time.time() - self.t0})

    def run(self) -> None:
        logger.info(f"AlphaEvolve - problem={self.problem.name}, iter={self.iterations}, output={self.output_dir}")
        b = self._eval(self.code)
        self.best = b.fitness
        logger.info(f"Baseline: {self.best:.4f}")

        for i in range(1, self.iterations + 1):
            self._iterate()
            if i % 100 == 0 or i == self.iterations:
                s = self.archive.get_stats()
                logger.info(f"Iter {i}/{self.iterations} best={self.best:.4f} snaps={s.total_snapshots} t={time.time()-self.t0:.1f}s")

        (self.output_dir / "best_code.py").write_text(self.code)
        self.archive.save(str(self.output_dir / "archive.json"))
        torch.save({"gen": self.gen, "best": self.best, "code": self.code, "hist": self.history, "prob": self.problem.name},
                   str(self.output_dir / "checkpoint.pt"))
        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        logger.info(f"Done! Best: {self.best:.4f}")

    def _load(self, path: str) -> None:
        c = torch.load(path)
        self.gen, self.best, self.code = c["gen"], c["best"], c["code"]
        self.history = c.get("hist", [])
        logger.info(f"Loaded checkpoint: {path}")


def main() -> int:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    p = argparse.ArgumentParser(description="AlphaEvolve Redesign")
    p.add_argument("--problem", default="steiner_tree", choices=list(PROBLEMS.keys()))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    prob = PROBLEMS[args.problem]
    try:
        if args.checkpoint:
            loop = EvolutionLoop(prob, args.iterations, args.output)
            loop._load(args.checkpoint)
        else:
            loop = EvolutionLoop(prob, args.iterations, args.output, seed=args.seed)
        loop.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted"); return 130
    except Exception as e:
        logger.exception(str(e)); return 1


if __name__ == "__main__":
    sys.exit(main())
