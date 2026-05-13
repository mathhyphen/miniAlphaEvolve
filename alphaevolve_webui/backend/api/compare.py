"""代码对比 API

提供两个版本代码的对比分析功能
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DiffLine(BaseModel):
    """差异行"""
    line_number_old: Optional[int]
    line_number_new: Optional[int]
    content: str
    change_type: str  # "added", "removed", "unchanged"


class CodeDiff(BaseModel):
    """代码差异"""
    old_version: str
    new_version: str
    diff_lines: List[DiffLine]
    summary: str


class CompareRequest(BaseModel):
    """代码对比请求"""
    version_a: str  # 版本 A 的代码
    version_b: str  # 版本 B 的代码
    language: Optional[str] = "python"


class CompareResponse(BaseModel):
    """代码对比响应"""
    version_a_length: int
    version_b_length: int
    diff: CodeDiff
    similarity: float  # 相似度 0.0 - 1.0


def _compute_diff(old_code: str, new_code: str) -> List[DiffLine]:
    """计算两个代码版本的差异

    Args:
        old_code: 旧版本代码
        new_code: 新版本代码

    Returns:
        差异行列表
    """
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    diff_lines = []

    # 简单的行对比算法
    max_lines = max(len(old_lines), len(new_lines))

    for i in range(max_lines):
        old_line = old_lines[i] if i < len(old_lines) else None
        new_line = new_lines[i] if i < len(new_lines) else None

        if old_line is None:
            diff_lines.append(
                DiffLine(
                    line_number_old=None,
                    line_number_new=i + 1,
                    content=new_line,
                    change_type="added",
                )
            )
        elif new_line is None:
            diff_lines.append(
                DiffLine(
                    line_number_old=i + 1,
                    line_number_new=None,
                    content=old_line,
                    change_type="removed",
                )
            )
        elif old_line != new_line:
            diff_lines.append(
                DiffLine(
                    line_number_old=i + 1,
                    line_number_new=i + 1,
                    content=new_line,
                    change_type="changed",
                )
            )
        else:
            diff_lines.append(
                DiffLine(
                    line_number_old=i + 1,
                    line_number_new=i + 1,
                    content=old_line,
                    change_type="unchanged",
                )
            )

    return diff_lines


def _compute_similarity(code_a: str, code_b: str) -> float:
    """计算两个代码版本的相似度

    Args:
        code_a: 代码 A
        code_b: 代码 B

    Returns:
        相似度分数 0.0 - 1.0
    """
    if not code_a and not code_b:
        return 1.0
    if not code_a or not code_b:
        return 0.0

    lines_a = set(code_a.splitlines())
    lines_b = set(code_b.splitlines())

    if not lines_a and not lines_b:
        return 1.0

    intersection = len(lines_a & lines_b)
    union = len(lines_a | lines_b)

    return intersection / union if union > 0 else 0.0


@router.post("")
async def compare_code(request: CompareRequest):
    """对比两个版本的代码

    Args:
        request: 包含两个版本代码的请求

    Returns:
        代码对比结果，包括差异和相似度
    """
    if not request.version_a.strip():
        raise HTTPException(status_code=400, detail="版本 A 代码不能为空")
    if not request.version_b.strip():
        raise HTTPException(status_code=400, detail="版本 B 代码不能为空")

    # 计算差异
    diff_lines = _compute_diff(request.version_a, request.version_b)

    # 计算相似度
    similarity = _compute_similarity(request.version_a, request.version_b)

    # 生成摘要
    added = sum(1 for d in diff_lines if d.change_type == "added")
    removed = sum(1 for d in diff_lines if d.change_type == "removed")
    changed = sum(1 for d in diff_lines if d.change_type == "changed")

    summary = f"新增 {added} 行，删除 {removed} 行，修改 {changed} 行"

    diff = CodeDiff(
        old_version="version_a",
        new_version="version_b",
        diff_lines=diff_lines,
        summary=summary,
    )

    return CompareResponse(
        version_a_length=len(request.version_a),
        version_b_length=len(request.version_b),
        diff=diff,
        similarity=similarity,
    )
