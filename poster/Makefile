.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=neurips-2026-poster -outdir=../output/pdf main.tex
clean:
	latexmk -c -jobname=neurips-2026-poster -outdir=../output/pdf main.tex
