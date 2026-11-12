.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=neurips-2026-poster -outdir=build main.tex
	cp build/neurips-2026-poster.pdf neurips-2026-poster.pdf
clean:
	latexmk -c -jobname=neurips-2026-poster -outdir=build main.tex
