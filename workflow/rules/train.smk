# Direct-run training experiments that don't go through a W&B sweep.
# Each rule produces artifacts under data/generated/<analysis>/ that plot
# rules consume. Outputs are wrapped in protected() when expensive to regenerate.


rule barrett_igr_figure2:
    input:
        script="analysis/barrett_igr_figure2/run.py",
    output:
        pt=protected("data/generated/barrett_igr_figure2/results.pt"),
        json="data/generated/barrett_igr_figure2/results.json",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--output-pt {output.pt} --output-json {output.json}"


rule barrett_igr_long_horizon:
    input:
        script="analysis/barrett_igr_long_horizon/run.py",
    output:
        synth_eta001=protected("data/generated/barrett_igr_long_horizon/synth_eta001.pt"),
        synth_eta003=protected("data/generated/barrett_igr_long_horizon/synth_eta003.pt"),
        mnist_tanh=protected("data/generated/barrett_igr_long_horizon/mnist_tanh_eta001.pt"),
        mnist_relu=protected("data/generated/barrett_igr_long_horizon/mnist_relu_eta001.pt"),
    params:
        out_dir="data/generated/barrett_igr_long_horizon",
    shell:
        "PYTHONPATH=src uv run python {input.script} --output-dir {params.out_dir}"


rule barrett_igr_trajectory:
    input:
        script="analysis/barrett_igr_trajectory/run.py",
    output:
        results=protected("data/generated/barrett_igr_trajectory/results.pt"),
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"


rule nonlinear_multi_geometry_suite:
    input:
        script="analysis/nonlinear_multi_geometry_suite/run.py",
    output:
        manifest=protected("data/generated/nonlinear_multi_geometry_suite/manifest.json"),
    params:
        out_dir="data/generated/nonlinear_multi_geometry_suite",
        fig_dir="results/figures/nonlinear_multi_geometry_suite",
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
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--output-dir {params.out_dir} --figure-dir {params.fig_dir} "
        "--manifest {output.manifest}"


rule nonlinear_power_retrain_geometry:
    input:
        script="analysis/nonlinear_power_retrain_geometry/run.py",
    output:
        results=protected("data/generated/nonlinear_power_retrain_geometry/results.json"),
    shell:
        "PYTHONPATH=src uv run python {input.script} --output {output.results}"
