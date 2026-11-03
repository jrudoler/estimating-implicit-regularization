# Historical validation-stopped ridge replacement, separate from Figure 5.
REPLACEMENT_CONDITIONS = {
    "target": (0.3, "none", 0., "none"),
    "ablated": (0., "none", 0., "none"),
    "ridge_T": (0., "ridge", 3.556925e-5, "T"),
    "ridge_T10x": (0., "ridge", 3.556925e-4, "control_10x"),
}
REPLACEMENT_SEEDS = [123, 42, 666, 314, 17, 111, 325, 643, 432, 51]

rule train_dropout_replacement:
    input:
        script="analysis/dropout_retrain_recovery/run.py",
        model="analysis/dropout_retrain_recovery/model.py",
        estimator="analysis/closed_form_dropout_sweep/run.py",
        core=["src/core/" + name + ".py" for name in ["models", "data", "bias", "estimators"]],
        images="data/raw/MNIST/raw/train-images-idx3-ubyte",
        labels="data/raw/MNIST/raw/train-labels-idx1-ubyte",
    output:
        protected("data/generated/dropout_retrain_recovery/results/{condition}_s{seed}.json"),
    wildcard_constraints:
        condition="target|ablated|ridge_T|ridge_T10x",
        seed="123|42|666|314|17|111|325|643|432|51",
    params:
        dropout=lambda w: REPLACEMENT_CONDITIONS[w.condition][0],
        family=lambda w: REPLACEMENT_CONDITIONS[w.condition][1],
        coef=lambda w: REPLACEMENT_CONDITIONS[w.condition][2],
        protocol=lambda w: REPLACEMENT_CONDITIONS[w.condition][3],
    resources:
        slurm_partition="whartonstat", runtime=240, mem_mb=16000, cpus_per_task=4,
    shell:
        "CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=4 .venv/bin/python {input.script:q} "
        "--tag {wildcards.condition}_s{wildcards.seed} --condition {wildcards.condition} "
        "--seed {wildcards.seed} --dropout {params.dropout} --penalty-family {params.family} "
        "--penalty-coef={params.coef} --protocol {params.protocol} "
        "--depth 3 --width 256 --lr 0.05 --momentum 0.9 --batch-size 2048 "
        "--max-epochs 500 --patience 50 --data-root data/raw "
        "--checkpoint-dir data/generated/dropout_retrain_recovery/checkpoints --output {output:q}"

rule summarize_dropout_replacement:
    input:
        script="analysis/summarize_dropout_replacement/run.py",
        results=expand("data/generated/dropout_retrain_recovery/results/{condition}_s{seed}.json",
                       condition=list(REPLACEMENT_CONDITIONS), seed=REPLACEMENT_SEEDS),
    output:
        "results/data/dropout_replacement_manuscript.json",
    shell:
        ".venv/bin/python {input.script:q} --inputs {input.results:q} --output {output:q}"
