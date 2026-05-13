"""演进控制 API

提供演进任务的启动、停止、状态查询和结果获取接口
支持 WebSocket 用于未来实时状态推送
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class EvolutionConfig(BaseModel):
    """演进配置"""
    problem_name: str
    max_iterations: int = 100
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8


class EvolutionResult(BaseModel):
    """演进结果"""
    iteration: int
    best_code: str
    best_score: float
    timestamp: str


class EvolutionStatus(BaseModel):
    """演进状态"""
    status: str  # "idle", "running", "paused", "completed", "error"
    current_iteration: int
    progress: float  # 0.0 - 1.0
    best_score: Optional[float] = None
    message: Optional[str] = None


# 模拟演进状态，实际应与演进引擎集成
_evolution_state: dict = {
    "status": "idle",
    "current_iteration": 0,
    "max_iterations": 100,
    "best_score": None,
    "message": None,
    "results": [],
}


@router.post("/start")
async def start_evolution(config: EvolutionConfig):
    """启动演进任务

    Args:
        config: 演进配置参数

    Returns:
        启动确认信息
    """
    if _evolution_state["status"] == "running":
        raise HTTPException(status_code=400, detail="演进任务已在运行中")

    _evolution_state["status"] = "running"
    _evolution_state["current_iteration"] = 0
    _evolution_state["max_iterations"] = config.max_iterations
    _evolution_state["best_score"] = None
    _evolution_state["message"] = f"演进任务已启动: {config.problem_name}"
    _evolution_state["results"] = []

    return {
        "status": "started",
        "message": f"演进任务已启动: {config.problem_name}",
        "config": config.model_dump(),
    }


@router.post("/stop")
async def stop_evolution():
    """停止当前演进任务

    Returns:
        停止确认信息
    """
    if _evolution_state["status"] != "running":
        raise HTTPException(status_code=400, detail="没有正在运行的演进任务")

    _evolution_state["status"] = "idle"
    _evolution_state["message"] = "演进任务已停止"

    return {
        "status": "stopped",
        "message": "演进任务已停止",
        "final_iteration": _evolution_state["current_iteration"],
    }


@router.get("/status", response_model=EvolutionStatus)
async def get_evolution_status():
    """获取当前演进状态

    Returns:
        当前演进状态信息
    """
    state = _evolution_state
    progress = (
        state["current_iteration"] / state["max_iterations"]
        if state["max_iterations"] > 0
        else 0.0
    )

    return EvolutionStatus(
        status=state["status"],
        current_iteration=state["current_iteration"],
        progress=progress,
        best_score=state["best_score"],
        message=state["message"],
    )


@router.get("/results")
async def get_evolution_results(limit: int = 50):
    """获取演进结果历史

    Args:
        limit: 返回结果数量限制，默认 50

    Returns:
        演进结果列表
    """
    results = _evolution_state["results"]
    return {
        "total": len(results),
        "results": results[-limit:] if len(results) > limit else results,
    }


# WebSocket 支持（预留接口，供未来实现实时推送）
@router.websocket("/ws/status")
async def websocket_status(websocket):
    """WebSocket 实时状态推送（预留）

    未来用于推送实时演进状态更新
    """
    # 预留接口，待实现
    pass
