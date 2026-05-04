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
        dgp="analysis/ols_dgp.py",
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
        dgp="analysis/ols_dgp.py",
    output:
        results="data/generated/linear_regression_ols/results.pt",
    resources:
        slurm_partition="whartonstat",
        runtime=120,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule linear_regression_ols_noisy:
    """Same DGP/training as linear_regression_ols but with noise_std=10 (vs default 1).
    Used as the baseline for the appendix bootstrap experiments: bootstrap recovery
    requires sufficient OLS sampling variance (which scales as sigma/sqrt(n)) for
    bootstrap-induced theta variability to span the symmetric-matrix space."""
    input:
        script="analysis/linear_regression_ols/run.py",
        dgp="analysis/ols_dgp.py",
    output:
        results="data/generated/linear_regression_ols_noisy/results.pt",
    resources:
        slurm_partition="whartonstat",
        runtime=120,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --noise-std 10.0 --output {output.results}"


rule ols_bootstrap_sigma_sweep:
    """Sweep observation-noise sigma at fixed t and m=num_endpoints, recording
    bootstrap full-matrix recovery distance to theory.  Powers Panel F of the
    appendix bootstrap figure.  Pure CPU."""
    input:
        script="analysis/ols_bootstrap_sigma_sweep/run.py",
        dgp="analysis/ols_dgp.py",
    output:
        results=protected("data/generated/ols_bootstrap_sigma_sweep/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
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
        "PYTHONPATH=src uv run python {input.script} {params.device} "
        "--train-samples 10000 --test-samples 5000 --probe-samples 2048 "
        "--widths 50,100,200,400,800,1600 "
        "--learning-rates 0.0005,0.001,0.005,0.01,0.05,0.1,0.5 "
        "--seeds 0 --epochs 50 --num-hidden-layers 5 "
        "--estimator-steps 5 --flow-k 10 "
        "--out {output.pt}"


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


rule ols_full_matrix_recovery:
    """Train 100 endpoints x 5 pools with per-endpoint early stopping (Panel D).
    Each endpoint stops when its own training loss plateaus, so stop steps vary.
    Pure CPU (10-d linear regression), so no GPU resources requested."""
    input:
        script="analysis/ols_full_matrix_recovery/run.py",
        dgp="analysis/ols_dgp.py",
    output:
        results=protected("data/generated/ols_full_matrix_recovery/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule ols_full_matrix_recovery_panel_b:
    """Train 100 endpoints x 5 pools with a fixed stop step equal to the
    canonical early-stopping epoch from the single-endpoint OLS run (Panel B).
    All endpoints share the same t so the stacked system Q theta_k = -g_k has
    a single well-defined Q, making the recovery exact in expectation.
    Pure CPU (10-d linear regression), so no GPU resources requested."""
    input:
        script="analysis/ols_full_matrix_recovery/run.py",
        dgp="analysis/ols_dgp.py",
        linear_data="data/generated/linear_regression_ols/results.pt",
    output:
        results=protected("data/generated/ols_full_matrix_recovery_panel_b/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
        mem_mb=8000,
        cpus_per_task=2,
    params:
        stop_step=lambda wildcards, input: __import__("torch").load(
            input.linear_data, weights_only=False
        )["config"]["stop_epoch"],
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results} "
        "--stop-step {params.stop_step}"


rule ols_bootstrap_recovery:
    """Bootstrap endpoint recovery: 100 resamples x 10 pools with per-endpoint
    early stopping (Panel D analog).  Each endpoint resamples rows of the
    original (X, y) with replacement rather than drawing a new ground-truth beta.
    Uses the noisier (sigma=3) baseline; bootstrap requires sufficient OLS
    sampling variance to be identifiable. Pure CPU (10-d linear regression)."""
    input:
        script="analysis/ols_bootstrap_recovery/run.py",
        linear_data="data/generated/linear_regression_ols_noisy/results.pt",
    output:
        results=protected("data/generated/ols_bootstrap_recovery/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
        mem_mb=8000,
        cpus_per_task=2,
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--linear-data {input.linear_data} --output {output.results}"


rule ols_bootstrap_recovery_panel_b:
    """Fixed-stop-step variant of the bootstrap recovery (used for Panels B and C).
    All bootstrap endpoints share the canonical stop step from the noisy single-
    endpoint baseline so Q_theory is well-defined. Pure CPU."""
    input:
        script="analysis/ols_bootstrap_recovery/run.py",
        linear_data="data/generated/linear_regression_ols_noisy/results.pt",
    output:
        results=protected("data/generated/ols_bootstrap_recovery_panel_b/results.pt"),
    resources:
        slurm_partition="whartonstat",
        runtime=60,
        mem_mb=8000,
        cpus_per_task=2,
    params:
        stop_step=lambda wildcards, input: __import__("torch").load(
            input.linear_data, weights_only=False
        )["config"]["stop_epoch"],
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--linear-data {input.linear_data} --output {output.results} "
        "--stop-step {params.stop_step}"

