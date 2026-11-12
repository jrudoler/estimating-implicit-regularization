# NeurIPS 2026 poster

Build from the repository root with `make -C poster`. The result is
`poster/neurips-2026-poster.pdf` (one page, 140 cm wide × 90 cm high).
Print at 100% / actual size, without fitting to a different paper size.

This uses the **main-conference** landscape size in the official instructions,
checked November 10, 2026:
https://neurips.cc/Conferences/2026/PosterInstructions
Workshop posters use a different format (61 × 85 cm portrait).

## Editing and dependencies

Edit `main.tex`. Positions are centimetres from the upper-left corner.
A TeX Live installation with latexmk, pdfLaTeX, BibTeX, natbib, cmbright, TikZ, multicol, and qrcode is
required. Body and math use Computer Modern Bright; existing vector figures
retain their original typography. The figure PDFs are bundled in `figures/`; no training, figure regeneration,
or files outside this folder are required. These are poster snapshots of the
workflow outputs in `results/figures/`, which remain the canonical analysis outputs.
To refresh a plot, rerun its workflow rule and copy the resulting PDF here.

The poster summarizes the current manuscript's endpoint gradient-matching
framework, elastic-net recovery, OLS early stopping, dropout ridge projection, and recovery
of Barrett and Dherin's implicit gradient regularization from trajectories.
The dropout text describes the scalar ridge estimate as a partial summary of the effect. Author order and acknowledgement come from `paper/main.tex`.
Public links come from the repository README. The manuscript is not modified.

Wharton Blue (#011F5B), Wharton Red (#990000), and Night Street (#2D2C41)
come from the supplied `wharton-colors.json`. Pale blue is a layout tint.
`assets/wharton-statistics.eps` is the supplied CMYK horizontal department logo;
its PDF is a vector conversion made with epstopdf. The logo is not redrawn.
The user's cmbright preference overrides using the supplied Whitney font.

A render preview is kept under `poster/build/qa/preview.png` for local visual QA;
it is not an additional analysis figure or print deliverable.

Numbered citations are built from `references.bib`, a compact selection from the
manuscript bibliography. The footer has three labeled QR codes linking directly to the paper, blog, and code repository.

Lucide icons (https://lucide.dev/icons/) identify the three main sections. Official SVGs are vendored in `assets/lucide/`, recolored
Wharton Blue, and converted to vector PDFs with `rsvg-convert`. The upstream
license is retained in that folder; rebuilding the poster needs no network or
SVG conversion.

## Folder contents

- `main.tex` and `references.bib`: editable source.
- `figures/`: vector figure snapshots used by the poster.
- `assets/`: Wharton branding and Lucide icons with license.
- `neurips-2026-poster.pdf`: compiled deliverable.
- `build/`: ignored LaTeX intermediates and local QA previews.

You can copy this entire folder elsewhere and build it with `make`.

The local `posterunsrtnat.bst` adapts `unsrtnat` to print surnames only and
use “et al.” for three or more authors, preserving full author metadata in
`references.bib`. References appear below future work in column three.

Typography uses 30/37 pt main text, 26/32 pt captions, navy section headings,
and dark ink body text. References and footer links use smaller dedicated styles.
