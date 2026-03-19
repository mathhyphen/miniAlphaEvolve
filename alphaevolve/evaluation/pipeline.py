"""Evaluation pipeline orchestrator."""

import logging
from typing import List, Optional, Dict, Any

from alphaevolve.core.data_structures import EvaluatorResult
from alphaevolve.evaluation.stages import EvaluationStage, StageResult

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Multi-stage evaluation pipeline with early termination.

    The pipeline runs evaluation stages in sequence, with the ability
    to terminate early if a stage fails. This saves computation by
    not running expensive stages on obviously bad candidates.

    Pipeline Flow:
        Stage 1 (Syntax) → Stage 2 (Basic Tests) → Stage 3 (Edge Cases) → Stage 4 (Performance)
              ↓                    ↓                      ↓                    ↓
           FAIL→return          FAIL→return           FAIL→return         Final result

    Usage:
        pipeline = EvaluationPipeline(stages=[
            SyntaxCheckStage(),
            BasicTestStage(test_cases, "func_name"),
            EdgeCaseStage(edge_cases, "func_name"),
            PerformanceStage(inputs, "func_name"),
        ], early_terminate=True)

        result = pipeline.evaluate(code)
    """

    def __init__(
        self,
        stages: List[EvaluationStage],
        early_terminate: bool = True,
        stage_weights: Optional[List[float]] = None,
    ) -> None:
        """Initialize evaluation pipeline.

        Args:
            stages: List of evaluation stages in order
            early_terminate: Whether to stop on first failure
            stage_weights: Optional weights for combining stage scores
        """
        self.stages = stages
        self.early_terminate = early_terminate
        self.stage_weights = stage_weights or [1.0] * len(stages)

        # Ensure weights match stages
        if len(self.stage_weights) != len(stages):
            raise ValueError("stage_weights must match number of stages")

    def add_stage(self, stage: EvaluationStage, index: int = -1) -> None:
        """Add evaluation stage to pipeline.

        Args:
            stage: Stage to add
            index: Position to insert (-1 for append)
        """
        if index < 0:
            self.stages.append(stage)
            self.stage_weights.append(1.0)
        else:
            self.stages.insert(index, stage)
            self.stage_weights.insert(index, 1.0)

    def evaluate(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluatorResult:
        """Run multi-stage evaluation.

        Args:
            code: Code to evaluate
            context: Optional context passed to stages

        Returns:
            EvaluatorResult with combined scores
        """
        stage_results: List[StageResult] = []
        total_time = 0.0
        context = context or {}

        for i, stage in enumerate(self.stages):
            logger.debug(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")

            result = stage.evaluate(code, context)
            stage_results.append(result)
            total_time += result.execution_time
            context[f"{stage.name}_result"] = result

            # Check for early termination
            if self.early_terminate and not result.passed:
                logger.debug(f"Early termination at stage {stage.name}")
                return self._create_final_result(
                    stage_results=stage_results,
                    total_time=total_time,
                    early_terminated=True,
                )

        # All stages completed
        return self._create_final_result(
            stage_results=stage_results,
            total_time=total_time,
            early_terminated=False,
        )

    def get_stage_names(self) -> List[str]:
        """Get list of stage names."""
        return [stage.name for stage in self.stages]

    def _create_final_result(
        self,
        stage_results: List[StageResult],
        total_time: float,
        early_terminated: bool,
    ) -> EvaluatorResult:
        """Create final EvaluatorResult from stage results.

        Args:
            stage_results: Results from each stage
            total_time: Total evaluation time
            early_terminated: Whether evaluation ended early

        Returns:
            Combined EvaluatorResult
        """
        if not stage_results:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                feedback="No stages executed",
                execution_time=total_time,
            )

        # Calculate weighted fitness
        total_weight = sum(
            w for r, w in zip(stage_results, self.stage_weights)
        )

        if total_weight > 0:
            weighted_fitness = sum(
                r.score * w for r, w in zip(stage_results, self.stage_weights)
            ) / total_weight
        else:
            weighted_fitness = 0.0

        # Determine if all stages passed
        all_passed = all(r.passed for r in stage_results)

        # Build feedback
        feedback_parts = [f"[{r.stage_name}] {r.feedback}" for r in stage_results]
        if early_terminated:
            feedback_parts.append("(early termination)")

        # Build metrics
        metrics = {
            "total_time": total_time,
            "stages_completed": len(stage_results),
            "total_stages": len(self.stages),
            "early_terminated": early_terminated,
        }

        for r in stage_results:
            for key, value in r.metrics.items():
                metrics[f"{r.stage_name}.{key}"] = value

        return EvaluatorResult(
            fitness=weighted_fitness,
            passed=all_passed,
            metrics=metrics,
            feedback="\n".join(feedback_parts),
            execution_time=total_time,
        )
