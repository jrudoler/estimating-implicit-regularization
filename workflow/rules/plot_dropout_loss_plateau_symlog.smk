"""Full-grid presentation variant; never trains or overwrites the linear figure."""
from pathlib import Path
from itertools import product

LOGPLATEAU_BASE = Path(workflow.basedir).resolve()
LOGPLATEAU_CODE = LOGPLATEAU_BASE.parents[1] if LOGPLATEAU_BASE.name == "rules" else LOGPLATEAU_BASE.parent
LOGPLATEAU_STUDY = "dropout_loss_plateau_absolute"
LOGPLATEAU_ROOT = "data/generated/" + LOGPLATEAU_STUDY
LOGPLATEAU_TAGS = [f"d{d}_w{w}_do{p}_s{s}" for d, w, p, s in product(
    (3, 5), (128, 256, 512), (0., .05, .1, .15, .2, .3, .4, .5),
    (123, 42, 666, 314, 17, 111, 325, 643, 432, 51))]


rule plot_dropout_loss_plateau_symlog:
    input:
        runs=expand(LOGPLATEAU_ROOT + "/{tag}/result.json", tag=LOGPLATEAU_TAGS),
        script=str(LOGPLATEAU_CODE / "analysis/plot_dropout_loss_plateau/run.py"),
        reporting=str(LOGPLATEAU_CODE / "src/core/loss_plateau_reporting.py"),
        style=str(LOGPLATEAU_CODE / "clean_fig.mplstyle"),
    output:
        figure="results/figures/dropout_bias_ridge_panel_loss_plateau_absolute_symlog.pdf",
        table="results/data/dropout_loss_plateau_absolute_grid_symlog.csv",
        receipt=LOGPLATEAU_ROOT + "/grid_validation_symlog.json",
    params:
        source=str(LOGPLATEAU_CODE / "src"),
        root=LOGPLATEAU_ROOT,
        study=LOGPLATEAU_STUDY,
    shell:
        "MPLCONFIGDIR=.snakemake/matplotlib PYTHONPATH={params.source:q} UV_NO_SYNC=1 "
        ".venv/bin/python {input.script:q} --results-dir {params.root:q} "
        "--study-name {params.study:q} --yscale symlog --figure {output.figure:q} "
        "--table {output.table:q} --receipt {output.receipt:q} --style {input.style:q}"
