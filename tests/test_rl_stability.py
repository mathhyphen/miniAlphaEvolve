"""Regression tests for RL/PPO stability fixes."""

import pytest

torch = pytest.importorskip("torch")

from alphaevolve.rl.patch_generator import CodeState, ExecutionResult, Patch, PatchGenerator, PatchSample, PatchType
from alphaevolve.rl.policy_network import PolicyConfig, PolicyNetwork
from alphaevolve.sandbox.executor import ExecutionResult as SandboxExecutionResult
from training.evolution_loop import EvolutionConfig, EvolutionLoop


def test_action_to_patch_accepts_all_policy_action_ids():
    generator = PatchGenerator(config=PolicyConfig(num_actions=64))
    code = "def solve(x):\n    return x\n"

    for action in range(generator.policy.cfg.num_actions):
        patch = generator.action_to_patch(action, code)
        assert patch.patch_type in list(PatchType)


def test_policy_network_handles_empty_code_masks_and_shape_validation():
    policy = PolicyNetwork(PolicyConfig(num_actions=8))
    code_tokens = torch.zeros((1, 8), dtype=torch.long)
    code_mask = torch.zeros((1, 8), dtype=torch.float32)

    logits, values = policy.forward(code_tokens, code_mask=code_mask)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()

    with pytest.raises(ValueError):
        policy.forward(
            torch.zeros((2, 8), dtype=torch.long),
            code_mask=torch.ones((2, 8), dtype=torch.float32),
            graph_features=torch.zeros((1, 64), dtype=torch.float32),
        )


def test_deterministic_actions_return_real_log_prob():
    policy = PolicyNetwork(PolicyConfig(num_actions=8))
    policy.eval()
    code_tokens = torch.ones((1, 8), dtype=torch.long)
    code_mask = torch.ones((1, 8), dtype=torch.float32)

    action, log_prob, _ = policy.get_action(code_tokens, code_mask=code_mask, deterministic=True)
    logits, _ = policy.forward(code_tokens, code_mask=code_mask)
    probs = torch.softmax(logits, dim=-1)
    expected = torch.distributions.Categorical(probs).log_prob(action)

    assert torch.allclose(log_prob, expected)


def test_evolution_loop_records_true_trajectory_states():
    class DummyProblem:
        initial_code = "def solve(n, edges, terminals):\n    return 1\n"
        function_name = "solve"

        @staticmethod
        def get_test_case():
            return {"args": (1, [], [])}

    loop = EvolutionLoop(DummyProblem(), EvolutionConfig(patches_per_episode=2))

    samples = [
        PatchSample(
            action=3,
            log_prob=-0.5,
            value=0.25,
            patch=Patch(
                patch_type=PatchType.REPLACE,
                target_line=1,
                content="    return 2",
                description="replace with two",
            ),
        ),
        PatchSample(
            action=4,
            log_prob=-0.25,
            value=0.75,
            patch=Patch(
                patch_type=PatchType.REPLACE,
                target_line=1,
                content="    return 3",
                description="replace with three",
            ),
        ),
    ]

    def fake_generate_patches(state, num_patches):
        del num_patches
        return [samples.pop(0)] if samples else []

    latencies = [10.0, 5.0]

    def fake_execute_patch(code, patch, problem):
        del code, patch, problem
        latency_ms = latencies.pop(0)
        return SandboxExecutionResult(
            success=True,
            output=latency_ms,
            error=None,
            execution_time=latency_ms / 1000.0,
            peak_memory_mb=0.0,
            exit_code=0,
        ), latency_ms

    loop._generate_patches = fake_generate_patches
    loop._execute_patch = fake_execute_patch

    episode = loop._run_episode(DummyProblem(), iteration=0)

    assert len(episode.trajectory) == 2
    assert episode.trajectory[0].state_code.endswith("return 1\n")
    assert episode.trajectory[1].state_code.endswith("return 2")
    assert episode.trajectory[0].action == 3
    assert episode.trajectory[0].old_log_prob == pytest.approx(-0.5)
    assert episode.trajectory[1].old_log_prob == pytest.approx(-0.25)


def test_patch_generator_reuses_the_same_optimizer_instance():
    generator = PatchGenerator(config=PolicyConfig(num_actions=8))
    state = CodeState(
        code="def solve(x):\n    return x + 1\n",
        execution_history=[ExecutionResult(passed=True, fitness=1.0, execution_time=0.01)],
    )
    optimizer_id = id(generator.optimizer)

    generator.update_policy(
        states=[state],
        actions=[0],
        rewards=[1.0],
        old_log_probs=[0.0],
        rollout_values=[0.0],
    )
    generator.update_policy(
        states=[state],
        actions=[0],
        rewards=[0.5],
        old_log_probs=[0.0],
        rollout_values=[0.0],
    )

    assert id(generator.optimizer) == optimizer_id
