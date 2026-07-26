# Figure-generation rules. One rule per plot entrypoint.
#
# Each rule:
#   - reads intermediates from data/generated/<analysis>/, including local
#     W&B run snapshots in runs.parquet
#   - writes the final PDF to results/figures/


rule plot_elasticnet_recovery:
    input:
        script="analysis/plot_elasticnet_recovery/run.py",
        style="clean_fig.mplstyle",
        runs="data/generated/elasticnet_train_and_recover/runs.parquet",
    output:
        pdf="results/figures/elasticnet_recovery_mean_se.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--runs-parquet {input.runs} "
        "--output {output.pdf}"


rule plot_dropout_bias_ridge_panel:
    input:
        script="analysis/plot_dropout_bias_ridge_panel/run.py",
        style="clean_fig.mplstyle",
        runs="data/generated/dropout_bias_estimation/runs.parquet",
    output:
        pdf="results/figures/dropout_bias_ridge_panel.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--runs-parquet {input.runs} "
        "--output {output.pdf}"


rule plot_closed_form_dropout_grid:
    input:
        script="analysis/plot_closed_form_dropout_grid/run.py",
        style="clean_fig.mplstyle",
        results="data/generated/closed_form_dropout_grid/results",
    output:
        pdf="results/figures/dropout_bias_ridge_panel_closed_form.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results-dir {input.results} "
        "--output {output.pdf}"


rule plot_dropout_trajectory_stability:
    input:
        script="analysis/plot_dropout_trajectory_stability/run.py",
        style="clean_fig.mplstyle",
        summary="data/generated/dropout_trajectory_lightning/summary.json",
    output:
        pdf="results/figures/dropout_lambda_trajectory_stability.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--summary {input.summary} "
        "--output {output.pdf}"


rule plot_scaling_data_vs_model:
    input:
        script="analysis/plot_scaling_data_vs_model/run.py",
        style="clean_fig.mplstyle",
    output:
        pdf="results/figures/scaling_data_vs_model.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--style {input.style} "
        "--output {output.pdf}"


rule plot_barrett_igr_figure2:
    input:
        script="analysis/plot_barrett_igr_figure2/run.py",
        style="clean_fig.mplstyle",
        results="data/generated/barrett_igr_figure2/results.pt",
    output:
        pdf="results/figures/barrett_igr_figure2.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--results {input.results} --out {output.pdf}"


rule plot_barrett_igr_long_horizon:
    input:
        script="analysis/plot_barrett_igr_long_horizon/run.py",
        style="clean_fig.mplstyle",
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


rule plot_method_vis:
    input:
        script="analysis/plot_method_vis/run.py",
        style="clean_fig.mplstyle",
    output:
        tradeoff="results/figures/tradeoff-vis.pdf",
        tradeoff_3d="results/figures/tradeoff-vis-3d.pdf",
        sgd_vs_fb="results/figures/sgd-vs-full-batch.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--out-tradeoff {output.tradeoff} "
        "--out-tradeoff-1d {output.tradeoff_3d} "
        "--out-sgd-vs-fb {output.sgd_vs_fb}"


rule plot_ols_composite:
    input:
        script="analysis/plot_ols_composite/run.py",
        style="clean_fig.mplstyle",
        linear_data="data/generated/linear_regression_ols/results.pt",
        panel_b_data="data/generated/ols_full_matrix_recovery_panel_b/results.pt",
        full_matrix_data="data/generated/ols_full_matrix_recovery/results.pt",
        lambda_epochs_data="data/generated/lambda_vs_epochs/results.pt",
    output:
        pdf="results/figures/ols_composite.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--linear-data {input.linear_data} "
        "--panel-b-data {input.panel_b_data} "
        "--full-matrix-data {input.full_matrix_data} "
        "--lambda-epochs-data {input.lambda_epochs_data} "
        "--out {output.pdf}"


rule plot_ols_bootstrap_composite:
    input:
        script="analysis/plot_ols_bootstrap_composite/run.py",
        style="clean_fig.mplstyle",
        linear_data="data/generated/linear_regression_ols_noisy/results.pt",
        bootstrap_panel_b_data="data/generated/ols_bootstrap_recovery_panel_b/results.pt",
        bootstrap_full_data="data/generated/ols_bootstrap_recovery/results.pt",
        sigma_sweep_data="data/generated/ols_bootstrap_sigma_sweep/results.pt",
    output:
        pdf="results/figures/ols_bootstrap_composite.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--linear-data {input.linear_data} "
        "--bootstrap-panel-b-data {input.bootstrap_panel_b_data} "
        "--bootstrap-full-data {input.bootstrap_full_data} "
        "--sigma-sweep-data {input.sigma_sweep_data} "
        "--out {output.pdf}"


rule plot_llm_ridge_model_grid:
    input:
        script="analysis/plot_llm_ridge_model_grid/run.py",
        style="clean_fig.mplstyle",
        model_grid="data/generated/olmo_ridge_estimation/model_grid",
    output:
        pdf="results/figures/llm_ridge_model_grid.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input-dir {input.model_grid} "
        "--output {output.pdf}"


rule plot_llm_ridge_bootstrap_grid:
    input:
        script="analysis/plot_llm_ridge_bootstrap_grid/run.py",
        style="clean_fig.mplstyle",
        bootstrap_grid="data/generated/olmo_ridge_estimation/bootstrap_grid",
    output:
        pdf="results/figures/llm_ridge_bootstrap_grid.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input-dir {input.bootstrap_grid} "
        "--output {output.pdf}"


rule plot_llm_ridge_chunk_grid:
    input:
        script="analysis/plot_llm_ridge_bootstrap_grid/run.py",
        style="clean_fig.mplstyle",
        chunk_grid="data/generated/olmo_ridge_estimation/chunk_grid",
    output:
        pdf="results/figures/llm_ridge_chunk_grid.pdf",
    shell:
        "PYTHONPATH=src uv run python {input.script} "
        "--input-dir {input.chunk_grid} "
        "--pattern '*_wikitext103_full_chunk10m.json' "
        "--output {output.pdf}"


rule stage_preserved_figure:
    input:
        src="data/provided/paper_assets/{fig_id}.{ext}",
    output:
        dst="results/figures/{fig_id}.{ext}",
    shell:
        "mkdir -p $(dirname {output.dst}) && cp {input.src} {output.dst}"
