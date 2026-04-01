"""问题管理 API

提供问题列表和问题详情查询接口
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ProblemBase(BaseModel):
    """问题基础信息"""
    name: str
    description: str
    benchmark: str
    difficulty: str


class ProblemDetail(ProblemBase):
    """问题详细信息"""
    constraints: List[str]
    evaluation_metric: str
    source_url: Optional[str] = None


class ProblemListItem(ProblemBase):
    """问题列表项"""
    id: str


# 模拟问题数据，实际应从数据库或配置加载
PROBLEMS_DATA = {
    "two_sum": ProblemDetail(
        name="两数之和",
        description="给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。",
        benchmark="LeetCode",
        difficulty="简单",
        constraints=["只存在一个有效答案", "不能重复使用同一元素", "数组长度 >= 2"],
        evaluation_metric="时间复杂度 O(n)，空间复杂度 O(1)",
        source_url="https://leetcode.cn/problems/two-sum/",
    ),
    "valid_parentheses": ProblemDetail(
        name="有效的括号",
        description="给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s，判断字符串是否有效。",
        benchmark="LeetCode",
        difficulty="简单",
        constraints=["空字符串视为有效", "必须成对出现", "顺序必须正确"],
        evaluation_metric="时间复杂度 O(n)，空间复杂度 O(n)",
        source_url="https://leetcode.cn/problems/valid-parentheses/",
    ),
    "merge_sorted_arrays": ProblemDetail(
        name="合并两个有序数组",
        description="给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。",
        benchmark="LeetCode",
        difficulty="简单",
        constraints=["初始化 nums1 和 nums2 的元素数量分别为 m 和 n", "可以假设 nums1 有足够空间保存 nums2 中的元素"],
        evaluation_metric="时间复杂度 O(m+n)，空间复杂度 O(1)",
        source_url="https://leetcode.cn/problems/merge-sorted-array/",
    ),
}


@router.get("", response_model=List[ProblemListItem])
async def get_problems():
    """获取所有可用问题列表

    Returns:
        问题列表
    """
    return [
        ProblemListItem(
            id=name,
            name=problem.name,
            description=problem.description,
            benchmark=problem.benchmark,
            difficulty=problem.difficulty,
        )
        for name, problem in PROBLEMS_DATA.items()
    ]


@router.get("/{name}", response_model=ProblemDetail)
async def get_problem_detail(name: str):
    """获取特定问题的详细信息

    Args:
        name: 问题名称/标识

    Returns:
        问题详细信息

    Raises:
        HTTPException: 当问题不存在时抛出 404 错误
    """
    if name not in PROBLEMS_DATA:
        raise HTTPException(status_code=404, detail=f"问题 '{name}' 不存在")
    return PROBLEMS_DATA[name]
