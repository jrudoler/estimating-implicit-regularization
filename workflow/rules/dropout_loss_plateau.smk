"""Loss-only stopping study; standalone default is three pilots, never the grid."""
from pathlib import Path
from itertools import product
import os
import json

PLATEAU_BASE = Path(workflow.basedir).resolve()
PLATEAU_CODE = PLATEAU_BASE.parents[1] if PLATEAU_BASE.name == "rules" else PLATEAU_BASE.parent
PLATEAU_STUDY = config.get("plateau_study", "dropout_loss_plateau")
if PLATEAU_STUDY not in ("dropout_loss_plateau", "dropout_loss_plateau_absolute"):
    raise ValueError("Unknown, undeclared loss-plateau protocol")
PLATEAU_ROOT = "data/generated/" + PLATEAU_STUDY
PLATEAU_RELATIVE = 0. if PLATEAU_STUDY.endswith("_absolute") else .001
PLATEAU_ABSOLUTE = 1e-7 if PLATEAU_STUDY.endswith("_absolute") else 0.
PLATEAU_FIGURE_NAME = "dropout_bias_ridge_panel_" + PLATEAU_STUDY.removeprefix("dropout_")
# The audited manuscript study must use the exact archived training sources.
if PLATEAU_STUDY.endswith("_absolute"):
    PLATEAU_CODE = Path(config.get("plateau_source", PLATEAU_ROOT + "/source-v1"))
PLATEAU_SCRATCH = config.get("plateau_scratch", os.environ.get("DROPOUT_NODE_SCRATCH", "/scratch/jrudoler/" + PLATEAU_STUDY))
PLATEAU_NODE = config.get("plateau_node", "node9")
PLATEAU_PILOTS = [f"d3_w256_do{p}_s123" for p in ("0.0", "0.3", "0.5")]
if PLATEAU_STUDY.endswith("_absolute"):
    PLATEAU_PILOTS += [f"d{d}_w{w}_do0.5_s123" for d, w in ((3, 128), (3, 512), (5, 128), (5, 512))]
PLATEAU_CELLS = list(product((3, 5), (128, 256, 512),
                            (0.0, .05, .1, .15, .2, .3, .4, .5),
                            (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)))
PLATEAU_TAGS = [f"d{d}_w{w}_do{p}_s{s}" for d, w, p, s in PLATEAU_CELLS]


rule download_manuscript_mnist:
    output:
        "data/raw/MNIST/raw/train-images-idx3-ubyte",
        "data/raw/MNIST/raw/train-labels-idx1-ubyte",
        "data/raw/MNIST/raw/t10k-images-idx3-ubyte",
        "data/raw/MNIST/raw/t10k-labels-idx1-ubyte",
    shell:
        ".venv/bin/python -c 'from torchvision.datasets import MNIST; "
        "MNIST(root=\"data/raw\", train=True, download=True); "
        "MNIST(root=\"data/raw\", train=False, download=True)'"


rule prepare_dropout_manuscript_source:
    input:
        archive="data/generated/dropout_loss_plateau_absolute/source-v1.tar.gz",
        script="analysis/prepare_dropout_source/run.py",
        manifest="config/dropout_source_files.json",
    output:
        expand("data/generated/dropout_loss_plateau_absolute/source-v1/{name}",
               name=json.loads(Path("config/dropout_source_files.json").read_text())),
    shell:
        ".venv/bin/python {input.script:q} --archive {input.archive:q} "
        "--manifest {input.manifest:q} --output-dir data/generated/dropout_loss_plateau_absolute/source-v1"


rule dropout_loss_plateau_pilots:
    input:
        expand(PLATEAU_ROOT + "/{tag}/result.json", tag=PLATEAU_PILOTS),


rule dropout_loss_plateau_full_grid:
    input:
        expand(PLATEAU_ROOT + "/{tag}/result.json", tag=PLATEAU_TAGS),


rule review_dropout_loss_plateau_pilots:
    input:
        runs=expand(PLATEAU_ROOT + "/{tag}/result.json", tag=PLATEAU_PILOTS),
        script=str(PLATEAU_CODE / "analysis/dropout_loss_plateau/review_pilot.py"),
    output:
        "results/data/" + PLATEAU_STUDY + "_pilot_review.json",
    params:
        source=str(PLATEAU_CODE / "src"),
        expanded="--expanded" if PLATEAU_STUDY.endswith("_absolute") else "",
    shell:
        "PYTHONPATH={params.source:q} .venv/bin/python {input.script:q} "
        "--runs {input.runs:q} --output {output:q} {params.expanded}"


rule plot_dropout_loss_plateau:
    input:
        runs=expand(PLATEAU_ROOT + "/{tag}/result.json", tag=PLATEAU_TAGS),
        script=str(PLATEAU_CODE / "analysis/plot_dropout_loss_plateau/run.py"),
        reporting=str(PLATEAU_CODE / "src/core/loss_plateau_reporting.py"),
        style=str(PLATEAU_CODE / "clean_fig.mplstyle"),
    output:
        figure="results/figures/" + PLATEAU_FIGURE_NAME + ".pdf",
        table="results/data/" + PLATEAU_STUDY + "_grid.csv",
        receipt=PLATEAU_ROOT + "/grid_validation.json",
    params:
        source=str(PLATEAU_CODE / "src"),
        study=PLATEAU_STUDY,
        root=PLATEAU_ROOT,
    shell:
        "MPLCONFIGDIR=.snakemake/matplotlib PYTHONPATH={params.source:q} .venv/bin/python {input.script:q} "
        "--results-dir {params.root:q} --study-name {params.study:q} --figure {output.figure:q} "
        "--table {output.table:q} --receipt {output.receipt:q} --style {input.style:q}"


rule approve_dropout_loss_plateau_grid:
    input:
        "results/data/" + PLATEAU_STUDY + "_pilot_review.json",
    output:
        PLATEAU_ROOT + "/pilot_gate_passed.txt",
    run:
        if not json.loads(Path(input[0]).read_text())["ready_for_grid_sensitivity_study"]:
            raise ValueError("Pilot robustness gate failed; refusing grid expansion")
        Path(output[0]).write_text("Pilot robustness gate passed.\n")


def plateau_source_input(relative: str) -> str:
    path = str(PLATEAU_CODE / relative)
    # The absolute-loss study uses immutable, archive-verified sources.
    # Restoring those sources must not invalidate completed training by mtime.
    return ancient(path) if PLATEAU_STUDY.endswith("_absolute") else path


rule train_dropout_loss_plateau:
    input:
        gate=lambda w: [] if f"d{w.depth}_w{w.width}_do{w.rate}_s{w.seed}" in PLATEAU_PILOTS
            else [ancient(PLATEAU_ROOT + "/pilot_gate_passed.txt")],
        script=plateau_source_input("analysis/dropout_loss_plateau/run.py"),
        publish=plateau_source_input("analysis/dropout_loss_plateau/publish.py"),
        storage=plateau_source_input("src/core/artifact_storage.py"),
        core=plateau_source_input("src/core/loss_plateau.py"),
        training=plateau_source_input("src/core/dropout_convergence.py"),
        import_helpers=plateau_source_input("src/core/dropout_continuation.py"),
        scalar=plateau_source_input("src/core/scalar_estimation.py"),
        model=plateau_source_input("src/core/models.py"),
        data=plateau_source_input("src/core/data.py"),
        lock=plateau_source_input("uv.lock"),
        images="data/raw/MNIST/raw/train-images-idx3-ubyte",
        labels="data/raw/MNIST/raw/train-labels-idx1-ubyte",
    output:
        result=protected(PLATEAU_ROOT + "/d{depth}_w{width}_do{rate}_s{seed}/result.json"),
        storage=protected(PLATEAU_ROOT + "/d{depth}_w{width}_do{rate}_s{seed}/storage.json"),
    wildcard_constraints:
        depth="3|5", width="128|256|512",
        rate="0[.](0|05|1|15|2|3|4|5)", seed="123|42|666|314|17|111|325|643|432|51",
    params:
        source=str(PLATEAU_CODE / "src"),
        directory=lambda w: f"{PLATEAU_ROOT}/d{w.depth}_w{w.width}_do{w.rate}_s{w.seed}",
        work=lambda w: f"{PLATEAU_SCRATCH or PLATEAU_ROOT}/d{w.depth}_w{w.width}_do{w.rate}_s{w.seed}",
        lock=PLATEAU_ROOT + "/.publish.lock",
        audit=lambda w: "--audit" if f"d{w.depth}_w{w.width}_do{w.rate}_s{w.seed}" in PLATEAU_PILOTS else "",
        study=PLATEAU_STUDY,
        relative=PLATEAU_RELATIVE,
        absolute=PLATEAU_ABSOLUTE,
    resources:
        slurm_partition="whartonstat", runtime=240, mem_mb=24000,
        cpus_per_task=2, gres="gpu:1", slurm_extra=f"--nodelist={PLATEAU_NODE}",
    shell:
        "PYTHONPATH={params.source:q} UV_NO_SYNC=1 .venv/bin/python {input.script:q} "
        "--depth {wildcards.depth} --width {wildcards.width} --dropout {wildcards.rate} "
        "--seed {wildcards.seed} --output-dir {params.work:q} {params.audit} "
        "--study-name {params.study:q} --relative-improvement {params.relative} --absolute-improvement {params.absolute} && "
        "PYTHONPATH={params.source:q} .venv/bin/python {input.publish:q} "
        "--source {params.work:q} --destination {params.directory:q} --lock {params.lock:q}"
