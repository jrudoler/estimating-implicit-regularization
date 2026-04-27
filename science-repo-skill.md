---
name: science-repository-layout
description: guidance for organizing computational science repositories around reproducible provenance, clear workflow structure, and separation of raw inputs, generated intermediates, reusable code, and final outputs. use when chatgpt / claude / agents needs to advise on repository structure, snakemake-based workflow layout, where files should live, how to add a new analysis step, or how to distinguish intermediate data from final scientific results.
---

> Note that you can use this as a Skill.md for your LLMs if you rename it to, say, Skill.md or CLAUDE.md or AGENTS.md or whatever....

# Science repository layout

Use this pattern to apply a consistent repository structure for computational science projects built around reproducibility and provenance.

The primary objective is to make it obvious:

* where inputs came from
* what generated an output
* which artifacts are intermediate
* which artifacts are final scientific results

## Core layout

Use this repository pattern:

```text
data/
  raw/
  provided/
  generated/
analysis/
  <rule_name>/run.py
src/
results/
  data/
  figures/
Snakefile
```

Interpret each directory as follows.

### `data/`

Use `data/` for data products. Do not put prose, manuscript text, or manuscript-ready figures here.

#### `data/raw/`

Store original downloaded or externally sourced inputs here.

Rules:

* preserve source form as closely as possible
* do not hand-edit files here
* use for external artifacts that should remain traceable to their origin

Examples:

* downloaded datasets
* raw parquet or csv exports
* externally provided network files

#### `data/provided/`

Store stable project-supplied inputs here.

Use this for inputs that are expected to exist as part of the repository or project environment, but are not necessarily downloaded by the workflow.

Examples:

* small reference tables
* metadata files
* curated annotations
* lookup tables
* stable configuration-like scientific assets

#### `data/generated/`

Store mechanically derived intermediate data products here.

Use this for artifacts produced by code from upstream inputs when the artifact is not itself a final scientific claim.

Examples:

* filtered datasets
* merged tables
* normalized representations
* preprocessed subgraphs

Decision rule:

* external source artifact to preserve: `data/raw/`
* stable project-supplied input: `data/provided/`
* code-generated intermediate: `data/generated/`

### `analysis/`

Use `analysis/` for workflow-step entrypoints.

Preferred pattern:

```text
analysis/<rule_name>/run.py
```

Each Snakemake rule should usually map to one folder in `analysis/` containing the minimal executable entrypoint for that step.

Put the following here:

* cli argument parsing
* loading explicit rule inputs
* calling shared code from `src/`
* writing explicit rule outputs

Do not let these folders accumulate reusable logic. Move shared code to `src/`.

Do not keep these in `analysis/` unless they are truly rule-local:

* duplicated helper functions
* reusable parsers
* shared plotting helpers
* common schemas
* generic utilities

### `src/`

Use `src/` as the shared Python package for the repository.

Put reusable logic here, including:

* file format parsing
* api access
* analysis utilities
* plotting defaults
* constants and filenames
* domain-specific shared code

Move code here whenever it would otherwise be duplicated across multiple `analysis/` scripts.

This separation keeps rule entrypoints small, shared logic testable, and refactors localized.

### `results/`

Use `results/` for outputs that are ready to inspect, cite, or include in a manuscript.

#### `results/data/`

Use for final tables intended for interpretation or downstream reading.

Examples:

* final summary csv files
* parquet tables used for interpretation
* manuscript-facing result tables

#### `results/figures/`

Use for final manuscript-facing figures.

Examples:

* pdf figures
* svg figures
* export-ready graphics

Decision rule:

* intermediate needed by another rule: `data/generated/`
* final output someone would open to understand the science: `results/`

## Workflow structure with Snakemake

Treat `Snakefile` as the workflow definition for the project.

Snakemake should be used to:

1. declare each analysis step as a rule
2. connect outputs of one step to inputs of the next
3. rebuild only what is stale or missing
4. provide a single reproducible execution path

Conceptual flow:

```text
raw or provided inputs
-> generated intermediates
-> final results tables and figures
```

## Execution conventions

Use these commands as defaults.

Run the workflow:

```bash
uv run snakemake --cores 1
```

Force one rule to rebuild:

```bash
uv run snakemake --cores 1 --forcerun <rule_name>
```

Run a rule script directly during development:

```bash
PYTHONPATH=src uv run python analysis/<rule_name>/run.py ...
```

Prefer Snakemake over direct script execution in normal use. Use direct execution mainly for debugging.

## Adding a new analysis step

When adding a new scientific step, follow this sequence:

1. decide whether the output is an intermediate or a final result
2. create `analysis/<rule_name>/run.py`
3. put reusable logic in `src/<package_name>/`
4. add the rule to `Snakefile`
5. write outputs to the correct location

Output placement rules:

* `data/generated/` for intermediates
* `results/data/` for final tables
* `results/figures/` for final figures

Preferred structure for each rule entrypoint:

* parse cli arguments
* read explicit inputs
* call shared library code
* write explicit outputs

Avoid hidden side effects. A rule should create the files it declares.

## Long-running jobs and `protected()`

Use Snakemake `protected()` for outputs that are expensive or painful to regenerate accidentally.

Example:

```python
rule expensive_step:
    input:
        "data/raw/input.parquet"
    output:
        protected("data/generated/expensive_result.parquet")
    shell:
        "uv run python analysis/expensive_step/run.py --input {input} --output {output}"
```

What `protected()` does:

* marks the output as write-protected after successful completion

Appropriate use cases:

* long-running preprocessing jobs
* large downloads
* computationally expensive model fits
* outputs with high accidental regeneration cost

Do not use `protected()` as a substitute for correct workflow design.

## Figure conventions

Figures should be generated programmatically and saved in publication-ready formats.

Prefer centralized matplotlib defaults rather than ad hoc restyling in each figure script.

Example:

```python
from matplotlib import pyplot as plt


def configure_matplotlib() -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Roboto"
    plt.rcParams["font.size"] = 10
```

Apply these figure principles:

* choose dimensions intentionally
* size the figure to its final rendered manuscript width
* for full-width manuscript figures, `figsize=(6.5, <height>)` is a common default
* for single-column figures, use the actual target column width instead of `6.5`
* keep text readable at final inclusion size
* use panel letters for multi-panel figures when conceptually appropriate
* rasterize only when vector density would be impractically large

Practical rule:

* use `6.5` inches as a reasonable default for full-width figures included directly at manuscript width
* do not make figures wider than their intended inclusion width and rely on later shrinking, because text will shrink too

The main principle is that figures should be ready to drop into the manuscript without manual editing.

## Quick placement guide

Use this mapping when deciding where something belongs:

* raw external inputs: `data/raw/`
* expected project-supplied inputs: `data/provided/`
* machine-generated intermediates: `data/generated/`
* workflow-step scripts: `analysis/<rule_name>/`
* shared Python utilities: `src/<package_name>/`
* final tables: `results/data/`
* final figures: `results/figures/`
* workflow wiring: `Snakefile`

## Operating philosophy

Maintain these separations:

* provenance from interpretation
* workflow steps from reusable code
* intermediates from final outputs

These separations matter because scientific repositories evolve constantly. If provenance is obvious from the repository structure, it becomes much easier to update analyses, regenerate figures, and defend scientific claims.

## Response behavior

When advising on a repository:

* recommend this layout unless the user has a strong existing convention that should be preserved
* explain file placement in terms of provenance and reproducibility, not aesthetics
* prefer minimal rule entrypoints and reusable code in `src/`
* distinguish intermediate artifacts from final outputs explicitly
* favor Snakemake as the primary execution interface when the project already uses it
* suggest `protected()` selectively for high-cost outputs
* recommend programmatic, publication-ready figure generation

When asked where a file or step should go, answer by assigning it to one of:

* `data/raw/`
* `data/provided/`
* `data/generated/`
* `analysis/<rule_name>/`
* `src/<package_name>/`
* `results/data/`
* `results/figures/`
* `Snakefile`

and justify the placement based on whether it is external input, stable provided input, generated intermediate, reusable code, workflow glue, or final output.


---

## Paper Submodules / Subdirectories

If using the optional "paper subdirectory" pattern, assume there is a paper manuscript in LaTeX format in `paper/`. 

You should generate figures in PDF format. Then as a final snakemake rule, you should migrate all paper-included assets (figures, tables in .tex format, etc) to `paper/generated/` with subdirectories:

* `paper/generated/figures/`
* `paper/generated/tables/`
* `paper/generated/_results.tex`

### Asset Moves

You can move assets to these folders using a rule like,

```snakemake
rule move_manifest:
    input:
        move_outputs=MOVE_ARTIFACTS,
        results_tex="paper/generated/_results.tex",
        script="analysis/tools/write_move_manifest.py",
    output:
        "paper/generated/manifest.json"
    shell:
        """
        mkdir -p paper/generated
        .venv/bin/python {input.script} \
          --output {output} \
          --inputs {input.move_outputs} {input.results_tex}
        """
```

The manifest.json can be used to track staleness etc.

### _results.tex

`_results.tex` is a latex-formatted file that enables you to write out values directly from code to your paper with `\newcommand` operators. For example,


```python
#!/usr/bin/env python3
"""Render claim result JSON files into LaTeX \\newcommand definitions."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

DIGIT_WORD = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def latex_escape(text: str) -> str:
    arrow_token = "ZZLRARROWZZ"
    text = text.replace("<->", arrow_token)

    escaped = text.replace("\\", "\\textbackslash{}")
    for a, b in [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]:
        escaped = escaped.replace(a, b)
    escaped = escaped.replace(arrow_token, "$\\leftrightarrow$")
    return escaped


def camel_token(token: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", token) if p]
    out = []
    for p in parts:
        normalized = "".join(DIGIT_WORD.get(ch, ch) for ch in p)
        if not normalized:
            continue
        out.append(normalized[:1].upper() + normalized[1:])
    return "".join(out)


def namespace_from_path(path: Path) -> str:
    # analysis/results/pinky/degree.json -> PinkyDegree
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        rel = parts[idx + 1 :]
    else:
        rel = parts
    stem_parts = list(rel[:-1]) + [Path(rel[-1]).stem]
    return "".join(camel_token(p) for p in stem_parts)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def sci_not_latex(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    # Keep at most 3 decimal places in the mantissa.
    mantissa_rounded = round(mantissa, 3)
    if abs(mantissa_rounded) >= 10:
        mantissa_rounded /= 10
        exponent += 1
    mantissa_str = f"{mantissa_rounded:.3f}".rstrip("0").rstrip(".")
    # don't do for 10^0:
    if exponent == 0:
        return mantissa_str
    # and for 10^1 just mult by 10, three dec, no x10:
    if exponent == 1:
        mantissa_str = f"{mantissa_rounded * 10:.3f}".rstrip("0").rstrip(".")
    return f"{mantissa_str} \\times 10^{{{exponent}}}"


def render_value(value: Any, key: str | None = None) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if key is not None and key.lower().endswith("percent") and is_number(value):
        return f"{float(value):,.2f}"
    if isinstance(value, int):
        return f"{value:,d}"
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return f"{int(value):,d}"
        return str(value)
    return latex_escape(str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lines: list[str] = []
    lines.append("% Auto-generated by analysis/tools/render_results_to_latex.py")
    lines.append("% Do not edit by hand.")

    for input_path in sorted(args.inputs):
        data = json.loads(input_path.read_text())
        namespace = namespace_from_path(input_path)
        lines.append(f"% Source: {input_path}")
        for key in sorted(data.keys()):
            value = data[key]
            if not is_scalar(value):
                lines.append(f"% Skipped non-scalar key: {key}")
                continue
            cmd = f"{namespace}{camel_token(key)}"
            lines.append(f"\\newcommand{{\\{cmd}}}{{{render_value(value, key)}}}")
            if is_number(value):
                sci_cmd = f"{cmd}SciNot"
                lines.append(
                    f"\\newcommand{{\\{sci_cmd}}}{{{sci_not_latex(float(value))}}}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
```

This creates a file that outputs both nonnumerical and numerical (in raw value and scientific notation) variables to your paper's `_results.tex` so that you can reference them in your paper like,


```latex
There were \NumberOfTotalCells cells in the dataset.
```

(You do not need to use the precise code above, but if you do, note that the scientific notation variables should be inside of mathmode brackets.)