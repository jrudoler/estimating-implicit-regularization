"""Read-only snapshot figure; never schedules training or stages manuscript files."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

INTERIM_BASE = Path(workflow.basedir).resolve()
INTERIM_CODE = INTERIM_BASE.parents[1] if INTERIM_BASE.name == "rules" else INTERIM_BASE.parent
INTERIM_STUDY = "dropout_loss_plateau_absolute"
INTERIM_ROOT = Path("data/generated") / INTERIM_STUDY
INTERIM_SNAPSHOT = config.get("preview_snapshot", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", INTERIM_SNAPSHOT):
    raise ValueError("preview_snapshot must be a UTC timestamp: YYYYMMDDTHHMMSSZ")
INTERIM_NAME = f"{INTERIM_STUDY}_interim_{INTERIM_SNAPSHOT}"
INTERIM_RECEIPT = INTERIM_ROOT / "previews" / f"{INTERIM_SNAPSHOT}.json"
# Once generated, retain the exact original input set even if more jobs finish.
if INTERIM_RECEIPT.exists():
    INTERIM_RUNS = [s["result"] for s in json.loads(INTERIM_RECEIPT.read_text())["sources"]]
else:
    INTERIM_RUNS = sorted(str(p) for p in INTERIM_ROOT.glob("*/result.json"))


rule plot_dropout_loss_plateau_interim:
    input:
        runs=INTERIM_RUNS,
        script=str(INTERIM_CODE / "analysis/plot_dropout_loss_plateau_interim/run.py"),
        reporting=str(INTERIM_CODE / "src/core/loss_plateau_reporting.py"),
        style=str(INTERIM_CODE / "clean_fig.mplstyle"),
    output:
        figure=f"results/figures/{INTERIM_NAME}.pdf",
        table=f"results/data/{INTERIM_NAME}.csv",
        receipt=str(INTERIM_RECEIPT),
    params:
        source=str(INTERIM_CODE / "src"),
        study=INTERIM_STUDY,
        snapshot=INTERIM_SNAPSHOT,
    shell:
        "MPLCONFIGDIR=.snakemake/matplotlib PYTHONPATH={params.source:q} UV_NO_SYNC=1 "
        ".venv/bin/python {input.script:q} --runs {input.runs:q} "
        "--study-name {params.study:q} --snapshot {params.snapshot:q} "
        "--figure {output.figure:q} --table {output.table:q} "
        "--receipt {output.receipt:q} --style {input.style:q}"
