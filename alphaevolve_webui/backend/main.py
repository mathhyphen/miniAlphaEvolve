"""FastAPI 应用入口

AlphaEvolve 演进系统后端服务
提供问题管理、演进控制、存档管理和代码对比等 API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import problems, evolution, archive, compare

app = FastAPI(
    title="AlphaEvolve API",
    description="AlphaEvolve 演进系统后端 API",
    version="1.0.0",
)

# 配置 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(problems.router, prefix="/api/problems", tags=["问题管理"])
app.include_router(evolution.router, prefix="/api/evolution", tags=["演进控制"])
app.include_router(archive.router, prefix="/api/archive", tags=["存档管理"])
app.include_router(compare.router, prefix="/api/compare", tags=["代码对比"])


@app.get("/")
async def root():
    """根路径，返回服务状态"""
    return {"status": "running", "service": "AlphaEvolve API"}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}
