# Interactive project page

A buildless, browser-local companion to *Estimating Implicit Regularization in Deep Learning*.

## Preview

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open http://127.0.0.1:8765/. Use HTTP rather than opening `index.html` directly: ES modules and the training worker require it. There are no npm dependencies or build steps. The notebook layout uses system fonts with no external font requests. No training data is transmitted.

## Validation

```sh
node --test site/tests/*.test.mjs
node --check site/app.mjs
node --check site/train-worker.mjs
```

The tests cover exact quadratic optima, constrained ridge projection, degenerate inputs, numerical finite-difference checks of the MLP gradients, reproducibility, and recovery across nine reference configurations (three paired data/initialization seeds × three penalties), plus forward/inverse UI regression tests. The recovery tests require <5% coefficient error and >95% gradient fit; the initial implementation achieved <0.1% coefficient error and >99% fit in those reference configurations. Random initializations are exploratory and may converge differently.

## GitHub Pages

The workflow at `../.github/workflows/pages.yml` validates and deploys this folder after relevant changes reach `main`. In repository **Settings → Pages → Build and deployment**, select **GitHub Actions** as the source. The expected project URL is:

https://jrudoler.github.io/estimating-implicit-regularization/

The workflow can be manually run on a feature branch to validate and upload an artifact; only `main` may deploy. It does not initialize the manuscript submodule or run scientific workflows. Relative asset paths support the repository subpath. Creating this branch does not publish the page.

## Files

- `index.html`: essay, accessible controls, methods, and research context.
- `style.css`: responsive layout and chart colors, reduced-motion support.
- `app.mjs`: canvas views, linked controls, and worker lifecycle.
- `math.mjs`: pure numerical functions, seeded data, explicit ReLU and tanh backpropagation.
- `train-worker.mjs`: full-batch training in interruptible chunks.
- `tests/math.test.mjs`: numerical regression tests.

## Scientific conventions

Every example uses `R(theta) = lambda * ||theta||^2`, so the gradient is `2 * lambda * theta`. The MLP has one hidden layer with selectable width (16, 32, or 64) and activation (ReLU or tanh), with 3 × width + 1 parameters, includes biases in the penalty, and uses mean squared error over 64 noisy synthetic observations. It trains with full-batch GD, learning rate 0.04 × 16 / width, for 20,000 steps. Diagnostics always use the unregularized, full-training-set gradient.

The scalar estimate is constrained to be nonnegative. Gradient explained is `1 - ||grad L + 2 lambda_hat theta||^2 / ||grad L||^2`, not prediction R-squared. A zero/near-zero denominator is displayed as unavailable. ReLU kinks and incomplete convergence can leave a residual. Recovery of an explicit penalty is a sanity check, not a demonstration that arbitrary implicit effects are equivalent to ridge. The final dropout discussion summarizes the manuscript; the live experiment does not train with dropout.

The 3D plot renders the actual two-parameter quadratic objective (unclipped wireframe with an automatically scaled height axis), not a slice through the MLP's parameter space. Both views use the same parameters and selected point. Gradient arrows are shown in the contour view; the surface labels the selected point and optimum. The selected point starts at the loss minimum. “Go to optimum” animates both views and disables at the optimum; changing the objective retains the selected point. Keyboard-accessible sliders provide alternatives to canvas pointer interactions.

Future extensions: early-stopped linear regression with its theoretical matrix regularizer; alternate candidate families; dropout and replacement training; downloadable experiment records and precomputed multi-seed paper results.

## Visual encoding

The notebook design follows Figure 1 (`analysis/plot_method_vis/run.py`): red loss, blue penalty, and a grayscale total objective. The total curve uses dark gray for contrast on white. Penalty curves and arrows are dashed; total curves are thicker. Circle/triangle/diamond markers, direct curve labels, and labeled arrows provide identification without color.

## Inverse experiment and initialization controls

Figure 2 fixes an observed scalar weight and fits a nonnegative ridge coefficient. Its animation interpolates to the analytic constrained least-squares solution, keeping the observation and plot/gradient scales fixed throughout. The error plot shows the squared remaining gradient versus lambda. The extended-observation checkbox includes impossible matches and the unidentifiable zero-weight case. Numerical state is kept independent of slider rounding.

Each figure has a Random initialization button:

- Figure 1: new weight, unchanged penalty.
- Figure 2: new candidate lambda, unchanged observed weight.
- Figure 3: new two-dimensional weight, unchanged objective and camera.
- Figure 4: new displayed network initialization seed, unchanged data seed and penalty; any active training is stopped and reset. Reset repeats that initialization, while Random initialization draws another one.

Figure 3 offers loss, penalty, and total layer checkboxes for both views, plus landscape opacity. Wireframes share a height scale across toggles. The gradient arrows always show positive (uphill) derivatives; tests verify finite differences and the canvas y-axis sign.
