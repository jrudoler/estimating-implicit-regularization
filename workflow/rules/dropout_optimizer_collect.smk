OPTIMIZER_FIX_SEEDS = (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)


def optimizer_fix_checkpoint_inputs(wildcards):
    import json
    sources = []
    for rate in (0.0, 0.3, 0.5):
        for seed in OPTIMIZER_FIX_SEEDS:
            record_path = Path(f"{AUDIT_ROOT}/results/d3_w256_do{rate}_s{seed}.json")
            record = json.loads(record_path.read_text())["trajectory"][-1]
            path = Path(record["provenance"]["checkpoint"])
            if not path.exists() and "/data/generated/" in str(path):
                path = Path("data/generated") / str(path).split("/data/generated/", 1)[1]
            sources.append(str(path))
    return sources


rule collect_optimizer_fix_stats:
    input:
        script="analysis/dropout_checkpoint_audit/confirm_optimizer_fix.py",
        audit="analysis/dropout_checkpoint_audit/run.py",
        scalar="src/core/scalar_estimation.py",
        model="src/core/models.py",
        data="src/core/data.py",
        results=[f"{AUDIT_ROOT}/results/d3_w256_do{rate}_s{seed}.json"
                 for rate in (0.0, 0.3, 0.5) for seed in OPTIMIZER_FIX_SEEDS],
        checkpoints=optimizer_fix_checkpoint_inputs,
        mnist_images="data/raw/MNIST/raw/train-images-idx3-ubyte",
        mnist_labels="data/raw/MNIST/raw/train-labels-idx1-ubyte",
    output:
        AUDIT_ROOT + "/optimizer_fix_stats.json",
    resources:
        slurm_partition="whartonstat", runtime=60, mem_mb=16000,
        cpus_per_task=2, slurm_extra="--gres=gpu:1",
    params:
        data=config.get("dropout_data_root", "data/raw"),
        device=config.get("device", "cuda"),
    shell:
        "PYTHONPATH=src uv run python {input.script} collect "
        "--results-dir data/generated/dropout_checkpoint_audit/results "
        "--data-root {params.data:q} --device {params.device} --output {output}"
