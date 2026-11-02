from itertools import product

# Explicit files ensure that changing any estimate invalidates the receipt/PDF.
DROPOUT_CELLS = list(product((3, 5), (128, 256, 512),
                            (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
                            (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)))
DROPOUT_JSONS = [f"data/generated/closed_form_dropout_grid/results/d{d}_w{w}_do{p}_s{s}.json"
                for d, w, p, s in DROPOUT_CELLS]

rule validate_dropout_grid:
    input:
        script="analysis/validate_dropout_grid/run.py",
        core="src/core/dropout_grid.py",
        scalar="src/core/scalar_estimation.py",
        results=DROPOUT_JSONS,
    output:
        "data/generated/closed_form_dropout_grid/validated.json",
    params:
        historical=config.get("dropout_historical_source", "locust:July-2026-closed-form-retrained-grid"),
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results-dir data/generated/closed_form_dropout_grid/results "
        "--historical-source {params.historical:q} --output {output}"


AUDIT_ROOT = "data/generated/dropout_checkpoint_audit"
AUDIT_TAGS = [f"d3_w256_do{p}_s{s}" for p in (0.0, 0.3, 0.5)
              for s in (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)]
AUDIT_SOURCE = Path(config.get("dropout_source_root", str(Path.cwd())))

if not config.get("dropout_recovered_only", False):
    include: "dropout_compute.smk"


localrules: confirm_optimizer_fix


if not config.get("optimizer_fix_stats_recovered_only", False):
    include: "dropout_optimizer_collect.smk"


rule confirm_optimizer_fix:
    input:
        script="analysis/dropout_checkpoint_audit/confirm_optimizer_fix.py",
        stats=AUDIT_ROOT + "/optimizer_fix_stats.json",
    output:
        "results/data/dropout_optimizer_fix_confirmation.json",
    shell:
        "PYTHONPATH=src uv run python {input.script} fit --stats {input.stats} --output {output}"


rule summarize_dropout_checkpoint_audit:
    input:
        script="analysis/dropout_checkpoint_audit/analyze.py",
        runs=expand(AUDIT_ROOT + "/results/{tag}.json", tag=AUDIT_TAGS),
    output:
        summary=AUDIT_ROOT + "/summary.json",
        table="results/data/dropout_checkpoint_controls.csv",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results-dir data/generated/dropout_checkpoint_audit/results "
        "--summary {output.summary} --table {output.table}"
