"""Standard dropout figure: validated, nonnegative closed-form estimates only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.plot_closed_form_dropout_grid.run import main

if __name__ == "__main__":
    if "--output" not in sys.argv:
        sys.argv.extend(["--output", str(Path(__file__).resolve().parents[2] / "results/figures/dropout_bias_ridge_panel.pdf")])
    main()
