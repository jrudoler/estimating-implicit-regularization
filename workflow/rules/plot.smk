# Figure-generation rules. One rule per plot entrypoint.
#
# Each rule:
#   - reads intermediates from data/generated/<analysis>/ or (for W&B-backed
#     plots) queries the sweep via config/sweeps.yaml
#   - writes the final PDF or PNG to results/figures/


rule plot_elasticnet_recovery:
    input:
        script="analysis/plot_elasticnet_recovery/run.py",
    output:
        pdf="results/figures/elasticnet_recovery_mean_se.pdf",
    params:
        sweep_id=lambda wc: config["elasticnet_train_and_recover"]["id"],
        entity_project=lambda wc: config["elasticnet_train_and_recover"]["entity_project"],
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--sweep-id {params.sweep_id} --entity-project {params.entity_project} "
        "--output {output.pdf}"


rule plot_dropout_bias_ridge_panel:
    input:
        script="analysis/plot_dropout_bias_ridge_panel/run.py",
    output:
        png="results/figures/dropout_bias_ridge_panel.png",
    params:
        sweep_id=lambda wc: config["dropout_bias_estimation"]["id"],
        entity=lambda wc: config["dropout_bias_estimation"]["entity_project"].split("/")[0],
        project=lambda wc: config["dropout_bias_estimation"]["entity_project"].split("/")[1],
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--sweep-id {params.sweep_id} --entity {params.entity} --project {params.project} "
        "--output {output.png}"


rule plot_barrett_igr_figure2:
    input:
        script="analysis/plot_barrett_igr_figure2/run.py",
        results="data/generated/barrett_igr_figure2/results.pt",
    output:
        pdf="results/figures/barrett_igr_figure2.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results {input.results} --out {output.pdf}"


rule plot_barrett_igr_long_horizon:
    input:
        script="analysis/plot_barrett_igr_long_horizon/run.py",
        synth_eta001="data/generated/barrett_igr_long_horizon/synth_eta001.pt",
        synth_eta003="data/generated/barrett_igr_long_horizon/synth_eta003.pt",
        mnist_tanh="data/generated/barrett_igr_long_horizon/mnist_tanh_eta001.pt",
        mnist_relu="data/generated/barrett_igr_long_horizon/mnist_relu_eta001.pt",
    output:
        pdf="results/figures/barrett_igr_long_horizon.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results {input.synth_eta001} {input.synth_eta003} {input.mnist_tanh} {input.mnist_relu} "
        "--out {output.pdf}"


rule plot_lambda_vs_epochs:
    input:
        script="analysis/plot_lambda_vs_epochs/run.py",
    output:
        pdf="results/figures/lambda_vs_epochs.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} --out {output.pdf}"


rule plot_method_vis:
    input:
        script="analysis/plot_method_vis/run.py",
    output:
        tradeoff="results/figures/tradeoff-vis.png",
        sgd_vs_fb="results/figures/sgd-vs-full-batch.png",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--out-tradeoff {output.tradeoff} --out-sgd-vs-fb {output.sgd_vs_fb}"


rule plot_linear_regression_ols:
    input:
        script="analysis/plot_linear_regression_ols/run.py",
    output:
        lambda_cmp="results/figures/Lambda_comparison-ols.pdf",
        pred_weights="results/figures/predictive_weights_comparison_ols.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--lambda-out {output.lambda_cmp} --weights-out {output.pred_weights}"


rule plot_ols_full_matrix_recovery:
    input:
        script="analysis/plot_ols_full_matrix_recovery/run.py",
        results="data/generated/ols_full_matrix_recovery/results.pt",
    output:
        recovery="results/figures/ols_full_matrix_recovery.pdf",
        distance="results/figures/ols_full_matrix_distance_to_theory.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input {input.results} "
        "--out-recovery {output.recovery} --out-distance {output.distance}"


rule stage_preserved_figure:
    input:
        src="data/provided/paper_assets/{fig_id}.{ext}",
    output:
        dst="results/figures/{fig_id}.{ext}",
    shell:
        "mkdir -p $(dirname {output.dst}) && cp {input.src} {output.dst}"
