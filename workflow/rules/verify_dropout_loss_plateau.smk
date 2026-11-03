"""CPU-only full-grid audit; run on the node retaining the trajectory checkpoints."""
from itertools import product
import os
from pathlib import Path

VERIFY_BASE = Path(workflow.basedir).resolve()
VERIFY_CODE = VERIFY_BASE.parents[1] if VERIFY_BASE.name == "rules" else VERIFY_BASE.parent
VERIFY_ROOT = "data/generated/dropout_loss_plateau_absolute"
VERIFY_CELLS = product((3, 5), (128, 256, 512), (0., .05, .1, .15, .2, .3, .4, .5),
                       (123, 42, 666, 314, 17, 111, 325, 643, 432, 51))
VERIFY_TAGS = [f"d{d}_w{w}_do{p}_s{s}" for d, w, p, s in VERIFY_CELLS]


rule verify_dropout_loss_plateau:
    input:
        runs=expand(VERIFY_ROOT + "/{tag}/result.json", tag=VERIFY_TAGS),
        storage=expand(VERIFY_ROOT + "/{tag}/storage.json", tag=VERIFY_TAGS),
        review="results/data/dropout_loss_plateau_absolute_pilot_review.json",
        tasks=str(VERIFY_CODE / "analysis/dropout_loss_plateau_absolute/grid_tasks.txt"),
        archive=VERIFY_ROOT + "/source-v1.tar.gz",
        script=str(VERIFY_CODE / "analysis/verify_dropout_loss_plateau/run.py"),
        core=str(VERIFY_CODE / "src/core/loss_plateau_verification.py"),
        reporting=str(VERIFY_CODE / "src/core/loss_plateau_reporting.py"),
        stability=str(VERIFY_CODE / "src/core/loss_plateau.py"),
    output:
        VERIFY_ROOT + "/full_grid_verification.json",
    params:
        source=str(VERIFY_CODE / "src"),
        root=VERIFY_ROOT,
        scratch=config.get("verification_scratch", config.get("plateau_scratch", os.environ.get("DROPOUT_NODE_SCRATCH", "/scratch/jrudoler/dropout_loss_plateau_absolute"))),
        node=config.get("verification_node", config.get("plateau_node", "node9")),
    resources:
        slurm_partition="whartonstat", runtime=120, mem_mb=8000, cpus_per_task=2,
        slurm_extra="--nodelist=" + config.get("verification_node", config.get("plateau_node", "node9")),
    shell:
        "PYTHONPATH={params.source:q} UV_NO_SYNC=1 .venv/bin/python {input.script:q} "
        "--results-dir {params.root:q} --scratch-root {params.scratch:q} --expected-node {params.node:q} "
        "--pilot-review {input.review:q} --task-file {input.tasks:q} --output {output:q}"
