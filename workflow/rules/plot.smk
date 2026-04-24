# Figure-generation rules. One rule per plot entrypoint.
#
# Each rule:
#   - reads intermediates from data/generated/<analysis>/
#   - writes the final PDF or PNG to results/figures/


rule plot_elasticnet_recovery:
    input:
        script="analysis/plot_elasticnet_recovery/run.py",
        runs="data/generated/elasticnet_train_and_recover/runs.parquet",
    output:
        pdf="results/figures/elasticnet_recovery_mean_se.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input {input.runs} --output-pdf {output.pdf}"


rule plot_dropout_bias_ridge_panel:
    input:
        script="analysis/plot_dropout_bias_ridge_panel/run.py",
        runs="data/generated/dropout_bias_estimation/runs.parquet",
    output:
        png="results/figures/dropout_bias_ridge_panel.png",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input {input.runs} --output-png {output.png}"


rule plot_barrett_igr_figure2:
    input:
        script="analysis/plot_barrett_igr_figure2/run.py",
        results="data/generated/barrett_igr_figure2/results.pt",
    output:
        pdf="results/figures/barrett_igr_figure2.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input {input.results} --output-pdf {output.pdf}"


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
        "--inputs {input.synth_eta001} {input.synth_eta003} {input.mnist_tanh} {input.mnist_relu} "
        "--output-pdf {output.pdf}"


rule plot_lambda_vs_epochs:
    input:
        script="analysis/plot_lambda_vs_epochs/run.py",
    output:
        pdf="results/figures/lambda_vs_epochs.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} --output-pdf {output.pdf}"


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
        "--out-lambda-cmp {output.lambda_cmp} --out-pred-weights {output.pred_weights}"


rule stage_preserved_figure:
    input:
        src="data/provided/paper_assets/{fig_id}.{ext}",
    output:
        dst="results/figures/{fig_id}.{ext}",
    shell:
        "mkdir -p $(dirname {output.dst}) && cp {input.src} {output.dst}"
