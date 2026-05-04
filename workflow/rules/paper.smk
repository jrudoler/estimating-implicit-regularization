# Paper assembly rules: stage final figures into paper/figures/
# and compile the PDF.


rule stage_paper_figure:
    input:
        src="results/figures/{fig_id}.{ext}",
    output:
        dst="paper/figures/{fig_id}.{ext}",
    shell:
        "mkdir -p $(dirname {output.dst}) && cp {input.src} {output.dst}"


rule stage_all_paper_figures:
    input:
        "paper/figures/tradeoff-vis.pdf",
        "paper/figures/sgd-vs-full-batch.pdf",
        "paper/figures/elasticnet_recovery_mean_se.pdf",
        "paper/figures/dropout_bias_ridge_panel.pdf",
        "paper/figures/barrett_igr_figure2.pdf",
        "paper/figures/barrett_igr_long_horizon.pdf",
        "paper/figures/ols_composite.pdf",
        "paper/figures/ols_bootstrap_composite.pdf",


rule paper_pdf:
    input:
        tex="paper/main.tex",
        figures=rules.stage_all_paper_figures.input,
    output:
        pdf="paper/main.pdf",
    shell:
        "cd paper && latexmk -pdf -interaction=nonstopmode main.tex"
