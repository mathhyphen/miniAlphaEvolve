"""FastAPI entry point for the AlphaEvolve product workbench."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alphaevolve.product import AlphaEvolveWorkbench
from alphaevolve_webui.backend.product_api import create_product_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AlphaEvolve Workbench API",
        description="AlphaEvolve-style algorithm discovery workbench API",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    workbench = AlphaEvolveWorkbench()
    app.include_router(create_product_router(workbench), prefix="/api", tags=["product"])

    @app.get("/")
    async def root():
        return {
            "status": "running",
            "service": "AlphaEvolve Workbench API",
            "version": "2.0.0",
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


def _cors_origins() -> list[str]:
    configured = os.environ.get(
        "ALPHAEVOLVE_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]


app = create_app()
