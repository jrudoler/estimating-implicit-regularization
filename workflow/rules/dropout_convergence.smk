import json

CONVERGENCE_ROOT = "data/generated/dropout_convergence"
CONVERGENCE_SOURCE = Path(config.get("convergence_source", "data/generated/dropout_checkpoint_audit/results"))

localrules: summarize_dropout_convergence


rule summarize_dropout_convergence:
    input:
        runs=expand(CONVERGENCE_ROOT + "/{tag}", tag=AUDIT_TAGS),
        script="analysis/dropout_convergence/summarize.py",
        grid="src/core/dropout_grid.py",
    output:
        "results/data/dropout_convergence_summary.json",
    shell:
        "PYTHONPATH=src uv run --frozen python {input.script} --runs {input.runs:q} --output {output:q}"


def convergence_source_record(wildcards):
    return str(CONVERGENCE_SOURCE / f"{wildcards.tag}.json")


def convergence_checkpoint(wildcards):
    source = json.loads(Path(convergence_source_record(wildcards)).read_text())
    endpoint = next(row for row in source["trajectory"] if row["fraction"] == 1.0)
    path = Path(endpoint["provenance"]["checkpoint"])
    if not path.exists() and "/data/generated/" in str(path):
        path = Path("data/generated") / str(path).split("/data/generated/", 1)[1]
    return str(path)


rule dropout_convergence:
    input:
        source=convergence_source_record,
        checkpoint=convergence_checkpoint,
        script="analysis/dropout_convergence/run.py",
        core="src/core/dropout_convergence.py",
        scalar="src/core/scalar_estimation.py",
        model="src/core/models.py",
        data="src/core/data.py",
        lock="uv.lock",
        images="data/raw/MNIST/raw/train-images-idx3-ubyte",
        labels="data/raw/MNIST/raw/train-labels-idx1-ubyte",
    output:
        directory(CONVERGENCE_ROOT + "/{tag}"),
    wildcard_constraints:
        tag="d3_w256_do(0.0|0.3|0.5)_s(123|42|666|314|17|111|325|643|432|51)",
    params:
        data="data/raw",
        epochs=config.get("convergence_epochs", 4000),
        interval=config.get("convergence_check_every", 250),
        lr=config.get("convergence_lr", .01),
        device=config.get("device", "cuda"),
    resources:
        slurm_partition="whartonstat", runtime=240, mem_mb=24000,
        cpus_per_task=2, slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run --frozen python {input.script} "
        "--source-result {input.source:q} --checkpoint {input.checkpoint:q} "
        "--data-root {params.data} --output-dir {output:q} "
        "--max-epochs {params.epochs} --check-every {params.interval} "
        "--lr {params.lr} --device {params.device}"
