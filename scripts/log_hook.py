#!/usr/bin/env python3
"""Log hook for recording experiment metrics in JSONL format."""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class LogHook:
    """Hook for logging experiment metrics to JSONL format."""

    def __init__(self, workspace_path: Path) -> None:
        """Initialize the log hook."""
        self.workspace_path = workspace_path
        self.log_dir = workspace_path / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "metrics.jsonl"

    def _get_git_metadata(self) -> Dict[str, Any]:
        """Best-effort git metadata lookup for the experiment workspace."""
        try:
            commit = subprocess.run(
                ["git", "-C", str(self.workspace_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "-C", str(self.workspace_path), "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(self.workspace_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            metadata = {"git_commit": commit, "git_dirty": bool(dirty)}
            if branch:
                metadata["git_branch"] = branch
            return metadata
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return {}

    def log(self, data: Dict[str, Any]) -> None:
        """Log a metric data point."""
        payload = dict(data)
        payload["_timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def log_environment(
        self,
        seed: Optional[int] = None,
        data_version: Optional[str] = None,
        config: Optional[Any] = None,
        config_path: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log environment information (system, Python, git, command)."""
        env_data = {
            "type": "environment",
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
        }
        env_data.update(self._get_git_metadata())
        if seed is not None:
            env_data["seed"] = seed
        if data_version is not None:
            env_data["data_version"] = data_version
        env_data["command"] = " ".join(sys.argv)
        if config_path is None and config is not None:
            config_path = config
        if config_path is not None:
            env_data["config"] = config_path
            env_data["config_path"] = config_path
        if extra:
            env_data.update(extra)
        self.log(env_data)

    def log_run_context(
        self,
        seed: int,
        data_version: Optional[str] = None,
        config_path: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log reproducibility metadata such as seed, data version, and config."""
        context: Dict[str, Any] = {"type": "run_context", "seed": seed}
        if data_version is not None:
            context["data_version"] = data_version
        if config_path is not None:
            context["config_path"] = config_path
        context.update(self._get_git_metadata())
        if extra:
            context.update(extra)
        self.log(context)

    def log_reproducibility(
        self,
        seed: int,
        data_version: Optional[str] = None,
        config: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Backward-compatible wrapper for older training scripts."""
        context: Dict[str, Any] = {"type": "reproducibility", "seed": seed}
        if data_version is not None:
            context["data_version"] = data_version
        if config is not None:
            context["config"] = config
        context.update(self._get_git_metadata())
        if extra:
            context.update(extra)
        self.log(context)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train",
    ) -> None:
        """Log epoch-level metrics."""
        data = {"epoch": epoch, "phase": phase, **metrics}
        self.log(data)
