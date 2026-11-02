# The summary-only import mode excludes these GPU producers entirely.
localrules: prepare_dropout_audit_specs

checkpoint prepare_dropout_audit_specs:
    input:
        script="analysis/dropout_checkpoint_audit/build_specs.py",
        grid=[str(AUDIT_SOURCE / f"data/generated/closed_form_dropout_grid/results/{tag}.json") for tag in AUDIT_TAGS],
        trajectories=[str(path) for tag in AUDIT_TAGS
                      if (path := AUDIT_SOURCE / f"data/generated/dropout_trajectory_lightning/results/{tag}.json").exists()],
    output:
        directory(AUDIT_ROOT + "/specs"),
    params:
        source=str(AUDIT_SOURCE),
    shell:
        "PYTHONPATH=src uv run python {input.script} --source-root {params.source:q} "
        "--output-root data/generated/dropout_checkpoint_audit"


def dropout_spec_path(wildcards):
    return str(Path(checkpoints.prepare_dropout_audit_specs.get().output[0]) / f"{wildcards.tag}.json")


def dropout_spec(wildcards):
    import json
    return json.loads(Path(dropout_spec_path(wildcards)).read_text())


def dropout_audit_inputs(wildcards):
    spec = dropout_spec(wildcards)
    if spec["checkpoints"]:
        paths = []
        for record in spec["checkpoints"]:
            path = Path(record["checkpoint"])
            if not path.exists() and "/data/generated/" in str(path):
                path = Path("data/generated") / str(path).split("/data/generated/", 1)[1]
            paths.append(str(path))
        return paths
    return [f"{AUDIT_ROOT}/replays/{wildcards.tag}.json",
            f"{AUDIT_ROOT}/checkpoints/{wildcards.tag}"]


rule dropout_checkpoint_replay:
    input:
        spec=dropout_spec_path,
        script="analysis/dropout_trajectory_lightning/run.py",
        models="src/core/models.py",
        data="src/core/data.py",
        dependencies="uv.lock",
    output:
        record=AUDIT_ROOT + "/replays/{tag}.json",
        checkpoints=directory(AUDIT_ROOT + "/checkpoints/{tag}"),
    wildcard_constraints:
        tag="d3_w256_do(0.0|0.3|0.5)_s[0-9]+",
    params:
        rate=lambda w: dropout_spec(w)["config"]["dropout"],
        seed=lambda w: dropout_spec(w)["config"]["seed"],
        epochs=lambda w: dropout_spec(w)["endpoint_epochs"],
        data=config.get("dropout_data_root", "data/raw"),
    resources:
        slurm_partition="whartonstat", runtime=180, mem_mb=16000,
        cpus_per_task=2, slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} --depth 3 --width 256 "
        "--dropout {params.rate} --seed {params.seed} --endpoint-epochs {params.epochs} "
        "--data-root {params.data:q} --checkpoint-dir {output.checkpoints} "
        "--output {output.record} --skip-estimation"


rule dropout_checkpoint_audit:
    input:
        spec=dropout_spec_path,
        checkpoints=dropout_audit_inputs,
        script="analysis/dropout_checkpoint_audit/run.py",
        scalar="src/core/scalar_estimation.py",
        models="src/core/models.py",
        data="src/core/data.py",
        dependencies="uv.lock",
    output:
        json=AUDIT_ROOT + "/results/{tag}.json",
    wildcard_constraints:
        tag="d3_w256_do(0.0|0.3|0.5)_s[0-9]+",
    resources:
        slurm_partition="whartonstat", runtime=180, mem_mb=16000,
        cpus_per_task=2, slurm_extra="--gres=gpu:1",
    params:
        data=config.get("dropout_data_root", "data/raw"),
        device=config.get("device", "cuda"),
    shell:
        "PYTHONPATH=src uv run python {input.script} --spec {input.spec} "
        "--output {output.json} --data-root {params.data:q} --device {params.device} --mask-draws 32"
