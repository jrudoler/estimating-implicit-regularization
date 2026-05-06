# Repository Agent Rules

This file is the canonical repo-local instructions for autonomous work in this repository.

## General Repo Boundaries

- `paper/` is a git submodule and a separate repository for the manuscript.
- `paper/` is available as read-only context for ordinary coding, methods, and experiment tasks.
- Treat the parent repo and `paper/` as operationally independent repos. Keep commits separate.
- Never edit files in `paper/` unless the user explicitly asks for manuscript, paper, or Overleaf changes.
- Default behavior for implementation, experiment, or methods work is to update code only, not `paper/`.
- The active manuscript location is `paper/`.

## Paper Repo Safety Rules

- Treat `paper/` as read-only by default, with one carveout: the Snakemake workflow writes paper-bound figures into `paper/figures/` via the `stage_paper_figure` rule. Files under `paper/figures/` are workflow output owned by the parent `Snakefile`, not hand-authored manuscript content.
- Hand-authored LaTeX (e.g. `paper/main.tex`, `paper/ref.bib`, `paper/macros.tex`) remains read-only unless the user explicitly requests manuscript edits.
- Before any manuscript edit, always `cd paper/` first.
- Before any manuscript edit, refresh `paper/` from the manuscript GitHub repo first.
- Treat the GitHub manuscript repo at `origin` as the default write target for manuscript changes.
- Assume Overleaf receives manuscript changes through GitHub synchronization.
- Do not use direct Overleaf git pulls, pushes, or remotes unless the user explicitly asks for an Overleaf-remote workflow.
- After any manuscript edit, always commit inside `paper/`.
- After any manuscript edit, always push from `paper/` to the manuscript GitHub repo.
- Keep `paper/` repo commits separate from parent repo commits.
- If `paper/` changes were made, remind the user that the parent repo's submodule pointer can be committed separately if they want the parent repo to reference the new manuscript commit.
- Never let unrelated code changes block normal manuscript syncing, and never let manuscript state block ordinary code work outside `paper/`.

## Practical Workflow

### Clone Or Update Submodules

```bash
git clone --recurse-submodules <PARENT_REPO_URL>
cd inductive-bias
```

```bash
git submodule update --init --recursive
```

```bash
git submodule update --init --recursive --remote paper
```

Inspect current manuscript remotes and submodule state:

```bash
git -C paper remote -v
git submodule status
```

### Refresh `paper/` Locally

```bash
cd paper
git fetch origin
git pull --ff-only origin main
cd ..
```

### Ordinary Code Work With `paper/` As Read-Only Context

```bash
git status --short
git -C paper status --short
# read files in paper/ for context, but do not stage or edit them
```

For implementation, methods, experiments, and autonomous findings, update code first; only edit the manuscript when the user explicitly asks for manuscript edits.

### Explicit Manuscript Edits In `paper/`

```bash
cd paper
git fetch origin
git pull --ff-only origin main
# make the requested manuscript edits only after the refresh succeeds
git status --short
cd ..
```

### Commit And Push Manuscript Changes To GitHub

```bash
cd paper
git add <FILES>
git commit -m "<MANUSCRIPT_COMMIT_MESSAGE>"
git push origin HEAD:main
cd ..
```

### Update The Parent Repo's Submodule Pointer

Run this only if you want the parent repo to record the new manuscript commit:

```bash
git add paper
git commit -m "Update paper submodule pointer"
```

## Analysis Output Convention

- For analysis figures, default to a single vector output format, `PDF`, unless the user explicitly asks for an additional raster export.
- Do not emit both `.png` and `.pdf` versions of the same analysis figure by default.
- Write figures to `results/figures/` via the corresponding `analysis/plot_<name>/run.py` rule in the Snakefile, not via ad-hoc scripts. The `stage_paper_figure` rule copies them into `paper/figures/`.
- Intermediates (`.pt`, `.parquet`, `.json` produced by training) go under `data/generated/<analysis>/`, not `results/`.
