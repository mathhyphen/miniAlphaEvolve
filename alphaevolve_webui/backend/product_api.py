"""Product API for the AlphaEvolve workbench."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from typing import Any

from pydantic import BaseModel, Field

from alphaevolve.product import AlphaEvolveWorkbench, EvaluationCase, TaskSpec


class RunRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=80)
    generations: int = Field(default=8, ge=1, le=100)
    archive_size: int = Field(default=8, ge=1, le=64)


class CustomCaseRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    args: list[Any] = Field(default_factory=list, max_length=20)
    expected: Any
    description: str = Field(default="", max_length=240)
    validator: str = Field(default="exact", pattern="^(exact|approx|contains)$")


class CustomRunRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=2000)
    initial_program: str = Field(min_length=1, max_length=20000)
    function_name: str = Field(default="solve", min_length=1, max_length=80)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    cases: list[CustomCaseRequest] = Field(min_length=1, max_length=30)
    generations: int = Field(default=4, ge=1, le=100)
    archive_size: int = Field(default=8, ge=1, le=64)


def create_product_router(workbench: AlphaEvolveWorkbench) -> APIRouter:
    router = APIRouter()

    @router.get("/tasks")
    async def list_tasks():
        return _ok([task.to_public_dict() for task in workbench.list_tasks()])

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        try:
            task = workbench.get_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _ok(task.to_public_dict())

    @router.post("/runs")
    async def start_run(request: RunRequest):
        try:
            run = workbench.start_run(
                task_id=request.task_id,
                generations=request.generations,
                archive_size=request.archive_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Model proposer failed. Check MINIMAX_API_KEY, "
                    "MINIMAX_MODEL, and MiniMax account access."
                ),
            ) from exc
        return _ok(run.to_dict())

    @router.post("/runs/custom")
    async def start_custom_run(request: CustomRunRequest):
        task = TaskSpec(
            task_id=f"custom_{_slug(request.title)}",
            title=request.title,
            objective=request.objective,
            initial_program=request.initial_program,
            function_name=request.function_name,
            constraints=tuple(request.constraints),
            tags=("custom", "user-defined"),
            cases=tuple(
                EvaluationCase(
                    case_id=case.case_id,
                    args=tuple(case.args),
                    expected=case.expected,
                    description=case.description,
                    validator=case.validator,
                )
                for case in request.cases
            ),
        )
        try:
            run = workbench.start_task_run(
                task=task,
                generations=request.generations,
                archive_size=request.archive_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Model proposer failed. Check MINIMAX_API_KEY, "
                    "MINIMAX_MODEL, and MiniMax account access."
                ),
            ) from exc
        return _ok(run.to_dict())

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            run = workbench.get_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _ok(run.to_dict())

    @router.get("/runs/{run_id}/archive")
    async def get_archive(run_id: str):
        try:
            run = workbench.get_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _ok([candidate.to_dict() for candidate in run.archive])

    @router.get("/runs/{run_id}/candidates/{candidate_id}")
    async def get_candidate(run_id: str, candidate_id: str):
        try:
            candidate = workbench.get_candidate(run_id, candidate_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _ok(candidate.to_dict())

    @router.get("/runs/{run_id}/export")
    async def export_best_program(run_id: str):
        try:
            program = workbench.export_best_program(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=program,
            media_type="text/x-python",
            headers={"Content-Disposition": f'attachment; filename="{run_id}-best.py"'},
        )

    return router


def _ok(data):
    return {"success": True, "data": data, "error": None}


def _slug(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return collapsed[:48] or "task"
