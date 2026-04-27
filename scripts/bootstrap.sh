#!/usr/bin/env bash
# Bootstrap the inductive-bias workflow: install uv, sync deps, prepare dirs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[bootstrap] uv: $(uv --version)"

echo "[bootstrap] syncing dependencies (including dev group)"
uv sync --group dev

echo "[bootstrap] snakemake: $(uv run snakemake --version)"

mkdir -p logs/slurm data/raw data/provided data/generated results/data results/figures
mkdir -p paper/generated/figures paper/generated/tables 2>/dev/null || true

echo "[bootstrap] done"
