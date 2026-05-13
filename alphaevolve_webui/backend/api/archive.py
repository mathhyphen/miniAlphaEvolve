"""存档管理 API

提供存档的保存、加载和列表查询功能
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ArchiveInfo(BaseModel):
    """存档信息"""
    id: str
    name: str
    description: Optional[str] = None
    created_at: str
    size: int  # 字节数


class SaveArchiveRequest(BaseModel):
    """保存存档请求"""
    name: str
    description: Optional[str] = None
    data: dict  # 存档数据


class LoadArchiveResponse(BaseModel):
    """加载存档响应"""
    id: str
    name: str
    data: dict
    loaded_at: str


# 模拟存档存储，实际应使用数据库
_archive_storage: dict = {}


@router.get("", response_model=List[ArchiveInfo])
async def get_archive_list():
    """获取存档列表

    Returns:
        所有存档的信息列表
    """
    archives = []
    for archive_id, archive in _archive_storage.items():
        archives.append(
            ArchiveInfo(
                id=archive_id,
                name=archive["name"],
                description=archive.get("description"),
                created_at=archive["created_at"],
                size=len(str(archive["data"])),
            )
        )
    return archives


@router.post("/save")
async def save_archive(request: SaveArchiveRequest):
    """保存存档

    Args:
        request: 保存存档请求，包含名称、描述和存档数据

    Returns:
        保存确认信息
    """
    archive_id = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _archive_storage[archive_id] = {
        "name": request.name,
        "description": request.description,
        "data": request.data,
        "created_at": datetime.now().isoformat(),
    }

    return {
        "status": "saved",
        "id": archive_id,
        "name": request.name,
        "created_at": _archive_storage[archive_id]["created_at"],
    }


@router.post("/load", response_model=LoadArchiveResponse)
async def load_archive(archive_id: str):
    """加载存档

    Args:
        archive_id: 存档标识

    Returns:
        存档数据

    Raises:
        HTTPException: 当存档不存在时抛出 404 错误
    """
    if archive_id not in _archive_storage:
        raise HTTPException(status_code=404, detail=f"存档 '{archive_id}' 不存在")

    archive = _archive_storage[archive_id]

    return LoadArchiveResponse(
        id=archive_id,
        name=archive["name"],
        data=archive["data"],
        loaded_at=datetime.now().isoformat(),
    )


@router.delete("/{archive_id}")
async def delete_archive(archive_id: str):
    """删除存档

    Args:
        archive_id: 存档标识

    Returns:
        删除确认信息

    Raises:
        HTTPException: 当存档不存在时抛出 404 错误
    """
    if archive_id not in _archive_storage:
        raise HTTPException(status_code=404, detail=f"存档 '{archive_id}' 不存在")

    del _archive_storage[archive_id]

    return {"status": "deleted", "id": archive_id}
