# Direct-run training experiments that don't go through a W&B sweep.
#
# Each rule produces artifacts under data/generated/<analysis>/. Resources
# declared here are consumed by the SLURM executor plugin when the workflow
# is run with `--profile workflow/profiles/slurm` (or `--executor slurm`):
#
#   slurm_partition -> -p / --partition
#   runtime         -> -t / --time (minutes)
#   mem_mb          -> --mem (MB)
#   cpus_per_task   -> -c / --cpus-per-task
#   slurm_extra     -> free-form sbatch flags (we use it for --gres=gpu:1)
#
# Under `--cores N` (local) these resources are ignored and the rule runs
# on the local machine.
#
# Device selection: the scripts autodetect CUDA > MPS > CPU. Override for
# a specific run with `--config device=<cpu|cuda|mps>`; the value threads
# into every training rule's shell via common.smk::device_arg().


rule lambda_vs_epochs:
    input:
        script="analysis/lambda_vs_epochs/run.py",
    output:
        results="data/generated/lambda_vs_epochs/results.pt",
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=16000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule linear_regression_ols:
    input:
        script="analysis/linear_regression_ols/run.py",
    output:
        results="data/generated/linear_regression_ols/results.pt",
    resources:
        slurm_partition="whartonstat",
        runtime=120,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule barrett_igr_figure2:
    input:
        script="analysis/barrett_igr_figure2/run.py",
    output:
        pt="data/generated/barrett_igr_figure2/results.pt",
        json="data/generated/barrett_igr_figure2/results.json",
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} {params.device} --out {output.pt}"


rule barrett_igr_long_horizon_synth_eta001:
    input:
        script="analysis/barrett_igr_long_horizon/run.py",
    output:
        pt="data/generated/barrett_igr_long_horizon/synth_eta001.pt",
        json="data/generated/barrett_igr_long_horizon/synth_eta001.json",
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=360,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--dataset synthetic --eta 0.01 --num-steps 50 --n-samples 500 --seed 0 "
        "--double-precision {params.device} --save {output.pt}"


rule barrett_igr_long_horizon_synth_eta003:
    input:
        script="analysis/barrett_igr_long_horizon/run.py",
    output:
        pt="data/generated/barrett_igr_long_horizon/synth_eta003.pt",
        json="data/generated/barrett_igr_long_horizon/synth_eta003.json",
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=360,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--dataset synthetic --eta 0.03 --num-steps 50 --n-samples 500 --seed 0 "
        "--double-precision {params.device} --save {output.pt}"


rule barrett_igr_long_horizon_mnist_tanh:
    input:
        script="analysis/barrett_igr_long_horizon/run.py",
    output:
        pt="data/generated/barrett_igr_long_horizon/mnist_tanh_eta001.pt",
        json="data/generated/barrett_igr_long_horizon/mnist_tanh_eta001.json",
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=360,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--dataset mnist --activation tanh --eta 0.01 --num-steps 30 --n-samples 1000 --seed 0 "
        "--double-precision {params.device} --save {output.pt}"


rule barrett_igr_long_horizon_mnist_relu:
    input:
        script="analysis/barrett_igr_long_horizon/run.py",
    output:
        pt="data/generated/barrett_igr_long_horizon/mnist_relu_eta001.pt",
        json="data/generated/barrett_igr_long_horizon/mnist_relu_eta001.json",
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=360,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--dataset mnist --activation relu --eta 0.01 --num-steps 30 --n-samples 1000 --seed 0 "
        "--double-precision {params.device} --save {output.pt}"


rule barrett_igr_trajectory:
    input:
        script="analysis/barrett_igr_trajectory/run.py",
    output:
        results=protected("data/generated/barrett_igr_trajectory/results.pt"),
    params:
        device=device_arg(),
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} {params.device} --output {output.results}"


rule nonlinear_multi_geometry_suite:
    input:
        script="analysis/nonlinear_multi_geometry_suite/run.py",
    output:
        manifest=protected("data/generated/nonlinear_multi_geometry_suite/manifest.json"),
    params:
        out_dir="data/generated/nonlinear_multi_geometry_suite",
        fig_dir="results/figures/nonlinear_multi_geometry_suite",
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--n-samples 256 --input-dim 12 --depth 2 --width 32 "
        "--max-epochs 350 --patience 60 --n-replicates 8 "
        "--estimation-max-epochs 2500 --estimation-patience 250 "
        "--target-gradient-scale 0.3 "
        "--output-dir {params.out_dir} --figure-dir {params.fig_dir} "
        "--manifest {output.manifest}"


rule nonlinear_multi_geometry_replicate_ablation:
    input:
        script="analysis/nonlinear_multi_geometry_replicate_ablation/run.py",
    output:
        manifest=protected("data/generated/nonlinear_multi_geometry_replicate_ablation/manifest.json"),
    params:
        out_dir="data/generated/nonlinear_multi_geometry_replicate_ablation",
        fig_dir="results/figures/nonlinear_multi_geometry_replicate_ablation",
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--output-dir {params.out_dir} --figure-dir {params.fig_dir} "
        "--manifest {output.manifest}"


rule ols_full_matrix_recovery:
    """Train 100 endpoints x 5 pools of OLS GD with callback early stopping;
    fit the symmetric Q via least squares; save tensors for the plot rule.
    Pure CPU (10-d linear regression), so no GPU resources requested."""
    input:
        script="analysis/ols_full_matrix_recovery/run.py",
    output:
        results=protected("data/generated/ols_full_matrix_recovery/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule nonlinear_power_retrain_geometry:
    input:
        script="analysis/nonlinear_power_retrain_geometry/run.py",
    output:
        results=protected("data/generated/nonlinear_power_retrain_geometry/results.json"),
    resources:
        slurm_partition="whartonstat",
        runtime=240,
        mem_mb=32000,
        cpus_per_task=4,
        slurm_extra="--gres=gpu:1",
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"
