# Paper assembly rules: stage final figures into paper/generated/figures/
# and compile the PDF.


rule stage_paper_figure:
    input:
        src="results/figures/{fig_id}.{ext}",
    output:
        dst="paper/generated/figures/{fig_id}.{ext}",
    shell:
        "mkdir -p $(dirname {output.dst}) && cp {input.src} {output.dst}"


rule stage_all_paper_figures:
    input:
        "paper/generated/figures/tradeoff-vis.png",
        "paper/generated/figures/sgd-vs-full-batch.png",
        "paper/generated/figures/elasticnet_recovery_mean_se.pdf",
        "paper/generated/figures/OLS_early_stopping_figure.pdf",
        "paper/generated/figures/Lambda_comparison-ols.pdf",
        "paper/generated/figures/predictive_weights_comparison_ols.pdf",
        "paper/generated/figures/lambda_vs_epochs.pdf",
        "paper/generated/figures/dropout_bias_ridge_panel.png",
        "paper/generated/figures/barrett_igr_figure2.pdf",
        "paper/generated/figures/barrett_igr_long_horizon.pdf",


rule paper_pdf:
    input:
        tex="paper/main.tex",
        figures=rules.stage_all_paper_figures.input,
    output:
        pdf="paper/main.pdf",
    shell:
        "cd paper && latexmk -pdf -interaction=nonstopmode main.tex"
