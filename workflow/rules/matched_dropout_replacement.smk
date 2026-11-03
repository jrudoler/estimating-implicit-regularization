"""Matched replacement of the final-grid dropout endpoints; no historical coefficients."""
MATCHED_SEEDS = [123, 42, 666, 314, 17, 111, 325, 643, 432, 51]
MATCHED_ROOT = "data/generated/dropout_matched_replacement"
MATCHED_SOURCE = "data/generated/dropout_loss_plateau_absolute/source-v1"
MATCHED_SCRATCH = config.get("matched_scratch", "/scratch/jrudoler/dropout_matched_replacement")

rule matched_dropout_replacement_all:
    input:
        expand(MATCHED_ROOT + "/{condition}_s{seed}/result.json",
               condition=["ablated", "ridge"], seed=MATCHED_SEEDS),

rule train_matched_dropout_replacement:
    input:
        script="analysis/matched_dropout_replacement/run.py",
        target="data/generated/dropout_loss_plateau_absolute/d3_w256_do0.3_s{seed}/result.json",
        baseline="data/generated/dropout_loss_plateau_absolute/d3_w256_do0.0_s{seed}/result.json",
        # Archive-verified immutable inputs: restoring the bundle must not
        # invalidate completed training merely by changing modification times.
        publish=ancient(MATCHED_SOURCE + "/analysis/dropout_loss_plateau/publish.py"),
        source=[ancient(MATCHED_SOURCE + "/" + p) for p in ["src/core/models.py", "src/core/data.py",
                "src/core/loss_plateau.py", "src/core/scalar_estimation.py",
                "src/core/dropout_continuation.py", "src/core/artifact_storage.py"]],
        raw=expand("data/raw/MNIST/raw/{name}", name=["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]),
    output:
        result=protected(MATCHED_ROOT + "/{condition}_s{seed}/result.json"),
        storage=protected(MATCHED_ROOT + "/{condition}_s{seed}/storage.json"),
    wildcard_constraints:
        condition="ablated|ridge", seed="123|42|666|314|17|111|325|643|432|51",
    params:
        source=MATCHED_SOURCE + "/src",
        scratch=lambda w: MATCHED_SCRATCH + f"/{w.condition}_s{w.seed}",
        destination=lambda w: MATCHED_ROOT + f"/{w.condition}_s{w.seed}",
        lock=MATCHED_ROOT + "/.publish.lock",
    resources:
        slurm_partition="whartonstat", runtime=60, mem_mb=24000, cpus_per_task=2,
        gres="gpu:1", slurm_extra="--nodelist=node9",
    log:
        MATCHED_ROOT + "/logs/{condition}_s{seed}.log",
    shell:
        "PYTHONPATH={params.source:q} OMP_NUM_THREADS=2 .venv/bin/python {input.script:q} "
        "--target {input.target:q} --baseline {input.baseline:q} --condition {wildcards.condition} "
        "--output-dir {params.scratch:q} > {log:q} 2>&1 && "
        "PYTHONPATH={params.source:q} .venv/bin/python {input.publish:q} "
        "--source {params.scratch:q} --destination {params.destination:q} --lock {params.lock:q} >> {log:q} 2>&1"

rule summarize_matched_dropout_replacement:
    input:
        script="analysis/matched_dropout_replacement/analyze.py",
        verifier="src/core/loss_plateau_verification.py",
        results=expand(MATCHED_ROOT + "/{condition}_s{seed}/result.json",
                       condition=["ablated", "ridge"], seed=MATCHED_SEEDS),
        references=expand("data/generated/dropout_loss_plateau_absolute/d3_w256_do{rate}_s{seed}/result.json",
                          rate=["0.0", "0.3"], seed=MATCHED_SEEDS),
    output:
        "results/data/dropout_matched_replacement.json",
    resources:
        slurm_partition="whartonstat", runtime=15, mem_mb=8000, cpus_per_task=2,
    shell:
        "PYTHONPATH=src .venv/bin/python {input.script:q} --inputs {input.results:q} --output {output:q}"
