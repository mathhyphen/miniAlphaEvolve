#!/usr/bin/env python3
"""Initialize an autoresearch-style experiment directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaevolve.autoresearch.runtime import RESULTS_HEADER, initialize_results_tsv


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates" / "autoresearch"
KIND_DIR = TEMPLATE_DIR / "kinds"


def render_template(src: Path, dest: Path, name: str) -> None:
    content = src.read_text(encoding="utf-8")
    content = content.replace("<name>", name)
    dest.write_text(content, encoding="utf-8")


def copy_optional_kind_files(kind: str, experiment_dir: Path, name: str) -> list[Path]:
    """Copy any extra files for a specific experiment kind."""

    kind_template_dir = KIND_DIR / kind
    if not kind_template_dir.exists():
        return []

    copied: list[Path] = []
    for src in kind_template_dir.rglob("*"):
        if not src.is_file():
            continue
        relative = src.relative_to(kind_template_dir)
        dest = experiment_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        render_template(src, dest, name)
        copied.append(dest)
    return copied


def create_experiment(
    name: str,
    workspace: Path | None = None,
    *,
    kind: str = "generic",
) -> Path:
    if workspace is None:
        workspace = BASE_DIR / "workspace"

    experiment_dir = workspace / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "artifacts").mkdir(exist_ok=True)

    render_template(TEMPLATE_DIR / "program.md", experiment_dir / "program.md", name)
    render_template(TEMPLATE_DIR / "memory.md", experiment_dir / "memory.md", name)
    render_template(TEMPLATE_DIR / "prepare.py", experiment_dir / "prepare.py", name)
    render_template(TEMPLATE_DIR / "candidate.py", experiment_dir / "candidate.py", name)
    copy_optional_kind_files(kind, experiment_dir, name)
    initialize_results_tsv(experiment_dir / "results.tsv")
    (experiment_dir / "artifacts" / ".gitkeep").write_text("", encoding="utf-8")
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize an autoresearch-style experiment")
    parser.add_argument("--name", required=True, help="Experiment directory name")
    parser.add_argument("--workspace", default=None, help="Optional workspace path")
    parser.add_argument(
        "--kind",
        default="generic",
        choices=["generic", "proof_search", "scientific_research"],
        help="Experiment kind template to use",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace) if args.workspace else None
    experiment_dir = create_experiment(args.name, workspace, kind=args.kind)

    print(f"Created autoresearch experiment: {experiment_dir}")
    print(f"Kind: {args.kind}")
    print("Files:")
    print(f"  - {experiment_dir / 'program.md'}")
    print(f"  - {experiment_dir / 'memory.md'}")
    print(f"  - {experiment_dir / 'prepare.py'}")
    print(f"  - {experiment_dir / 'candidate.py'}")
    print(f"  - {experiment_dir / 'results.tsv'}")
    print("")
    print("Next step:")
    print(f"  python -m scripts.run_autoresearch {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
