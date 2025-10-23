"""Utilities for inspecting locally cached Weights & Biases sweeps.

These helpers are designed for offline W&B runs stored inside the repository,
making it easier to discover run directories, collect summaries, and locate
artifacts when generating analysis notebooks or plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WANDB_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("logs") / "wandb" / "wandb",
    Path("logs") / "wandb",
    Path("wandb"),
)

RUN_DIR_PATTERNS: tuple[str, ...] = ("run-*-{run_id}", "offline-run-*-{run_id}")

__all__ = [
    "find_wandb_base_dirs",
    "get_sweep_directory",
    "list_sweep_run_ids",
    "get_sweep_run_paths",
    "load_run_summary",
    "load_run_config",
    "load_sweep_summaries",
    "load_sweep_configs",
    "collect_sweep_artifacts",
]


def find_wandb_base_dirs(
    repo_root: Path | None = None,
    extra_dirs: Iterable[Path] | None = None,
) -> list[Path]:
    """Return all existing W&B directories under the repository.

    Parameters
    ----------
    repo_root:
        Optional repository root. Defaults to the parent of this file.
    extra_dirs:
        Additional directories to probe, useful when experiments use a custom
        ``WANDB_DIR`` override.
    """
    base = repo_root or REPO_ROOT
    candidates: list[Path] = []
    if extra_dirs is not None:
        candidates.extend(extra_dirs)
    candidates.extend(base / rel for rel in DEFAULT_WANDB_RELATIVE_PATHS)

    existing = []
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else base / candidate
        if path.exists():
            existing.append(path.resolve())
        else:
            LOGGER.debug("Skipping missing W&B path: %s", path)
    if not existing:
        LOGGER.warning("No W&B directories found beneath %s", base)
    return existing


def get_sweep_directory(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> Path | None:
    """Locate the directory that caches metadata for ``sweep_id``."""
    for base_dir in find_wandb_base_dirs(repo_root=repo_root, extra_dirs=extra_dirs):
        sweep_dir = base_dir / f"sweep-{sweep_id}"
        if sweep_dir.exists():
            return sweep_dir
    LOGGER.warning("Unable to locate sweep %s under known W&B directories", sweep_id)
    return None


def list_sweep_run_ids(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> list[str]:
    """List all run identifiers associated with ``sweep_id``."""
    sweep_dir = get_sweep_directory(sweep_id, repo_root=repo_root, extra_dirs=extra_dirs)
    if sweep_dir is None:
        return []

    run_ids = []
    for config_path in sweep_dir.glob("config-*.yaml"):
        run_id = config_path.stem.removeprefix("config-")
        if run_id:
            run_ids.append(run_id)
    if not run_ids:
        LOGGER.warning("Sweep %s has no run config files at %s", sweep_id, sweep_dir)
    run_ids.sort()
    return run_ids


def _resolve_run_directory(
    run_id: str,
    base_dirs: Sequence[Path],
) -> Path | None:
    for base_dir in base_dirs:
        for pattern in RUN_DIR_PATTERNS:
            for candidate in base_dir.glob(pattern.format(run_id=run_id)):
                if candidate.is_dir():
                    return candidate.resolve()
    return None


def get_sweep_run_paths(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> dict[str, Path]:
    """Return a mapping from run id to its local directory for ``sweep_id``."""
    base_dirs = find_wandb_base_dirs(repo_root=repo_root, extra_dirs=extra_dirs)
    if not base_dirs:
        return {}

    run_paths: dict[str, Path] = {}
    missing: list[str] = []
    for run_id in list_sweep_run_ids(sweep_id, repo_root=repo_root, extra_dirs=extra_dirs):
        resolved = _resolve_run_directory(run_id, base_dirs)
        if resolved is None:
            missing.append(run_id)
        else:
            run_paths[run_id] = resolved

    if missing:
        LOGGER.warning("Found config files for runs without directories: %s", ", ".join(missing))
    return dict(sorted(run_paths.items()))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        LOGGER.debug("Missing JSON file: %s", path)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse JSON at %s: %s", path, exc)
    return {}


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    """Load the ``wandb-summary.json`` payload for a given run directory."""
    summary_path = run_dir / "files" / "wandb-summary.json"
    if not summary_path.exists():
        LOGGER.warning("Run %s has no wandb-summary.json", run_dir)
        return {}
    return _load_json(summary_path)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load the W&B ``config.yaml`` for ``run_dir``."""
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        LOGGER.warning("Run %s has no config.yaml", run_dir)
        return {}

    if yaml is None:  # pragma: no cover - runtime guard
        LOGGER.error("PyYAML is not installed; cannot parse %s", config_path)
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # type: ignore[union-attr]
        LOGGER.error("Failed to parse YAML config at %s: %s", config_path, exc)
        return {}
    return data if isinstance(data, Mapping) else {}


def load_sweep_summaries(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect summaries for every run in ``sweep_id``."""
    run_paths = get_sweep_run_paths(sweep_id, repo_root=repo_root, extra_dirs=extra_dirs)
    summaries = {run_id: load_run_summary(path) for run_id, path in run_paths.items()}
    return summaries


def load_sweep_configs(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect W&B configs for every run in ``sweep_id``."""
    run_paths = get_sweep_run_paths(sweep_id, repo_root=repo_root, extra_dirs=extra_dirs)
    configs = {run_id: load_run_config(path) for run_id, path in run_paths.items()}
    return configs


def collect_sweep_artifacts(
    sweep_id: str,
    repo_root: Path | None = None,
    *,
    extra_dirs: Iterable[Path] | None = None,
    pattern: str = "**/*",
) -> dict[str, list[Path]]:
    """List artifact files beneath ``files/artifacts`` for each run in a sweep."""
    run_paths = get_sweep_run_paths(sweep_id, repo_root=repo_root, extra_dirs=extra_dirs)
    artifact_map: dict[str, list[Path]] = {}

    for run_id, run_dir in run_paths.items():
        artifact_root = run_dir / "files" / "artifacts"
        if not artifact_root.exists():
            LOGGER.debug("Run %s has no artifacts directory at %s", run_id, artifact_root)
            artifact_map[run_id] = []
            continue

        artifacts = [
            path
            for path in artifact_root.glob(pattern)
            if path.is_file()
        ]
        artifact_map[run_id] = sorted(artifacts)
    return artifact_map

