# SWA on an MLP: effective-regularizer identification

Date: 2026-07-26

## Executive summary

Reviewer 2 asks for one application where the implicit regularizer is not known
in advance, suggesting mixup, SWA, or learning-rate warmup. SWA is a good test
case, but only if its two ingredients are separated:

1. a modified, relatively high learning-rate trajectory;
2. averaging the weights along that trajectory.

Calling the difference between ordinary decayed SGD and the final SWA weights
"the effect of averaging" would confound these components. We therefore branch
three conditions from one shared epoch-100 MNIST MLP checkpoint:

- ordinary decayed-SGD tail;
- constant-high-LR tail, using its last iterate;
- the equal average of all 100 constant-high-LR tail endpoints.

Both tails use identical minibatch orders. The MLP has depth 3, width 256, no
dropout or BatchNorm, and is evaluated on its training loss. Results use the
paper's ten standard seeds.

The averaging operator has a clear effective regularizer:

```text
R_SWA(theta) = lambda * ||theta - theta_0||_2^2,
```

where `theta_0` is the checkpoint at which averaging begins. This is a temporal
or checkpoint-centered quadratic, not ordinary weight decay toward the origin.
With the convention above, the closed-form endpoint coefficient is

```text
lambda_hat = 0.006911 +/- 0.000015.
```

At the averaged endpoint, this family has mean gradient cosine
`0.961 +/- 0.002` and explains `92.3% +/- 0.5%` of the loss-gradient energy.
Ordinary origin-centered L2 explains only `27.7% +/- 0.4%`.

The causal displacement check is even stronger. For the difference between the
last high-LR iterate and its weight average,

```text
delta_avg = theta_SWA - theta_last,
```

the direction `-grad R_SWA(theta_last)` has cosine `0.999` and explains
`99.7%` of squared displacement. Origin-centered L2 explains `31.2%`.

Finally, replacing averaging with the fitted explicit checkpoint-centered
penalty recovers the SWA solution:

- recovered displacement cosine from `theta_0`: `0.99931 +/- 0.00004`;
- parameter distance to SWA, normalized by the size of the SWA effect:
  `0.11085 +/- 0.00120`;
- the unregularized high-LR last iterate is `0.87824 +/- 0.00332` SWA-effect
  units away;
- training-set prediction agreement with SWA: `100%` in every seed.

This is strong evidence that, locally within the late-training basin, SWA
averaging behaves like a proximal L2 penalty anchored at the start of averaging.

## Why the ablation matters

SWA is not itself a gradient added at every optimization step. It is the map

```text
(theta_101, ..., theta_200) -> mean_t theta_t.
```

Direct endpoint matching can still ask which explicit regularizer would make the
averaged weights stationary, but that alone does not isolate the averaging
operator from the trajectory that generated the weights. The displacement
analysis supplies the missing ablation:

```text
theta_SWA - theta_last ~= -a * grad P(theta_last).
```

The coefficient `a` confounds a penalty strength with an undefined effective
step size, so the displacement cosine and projection R2 are the interpretable
quantities. They show that averaging moves almost exactly down the gradient of
the checkpoint-centered quadratic.

The geometry is intuitive. Over this late-training window, the trajectory is
nearly a ray leaving `theta_0`. Its average lies partway along that ray, so
averaging pulls the last iterate back toward `theta_0`. The endpoint
gradient-matching result adds the nontrivial statement that the loss gradient at
the average is also balanced by that same proximal force.

## Why "full SWA versus decayed SGD" has no single positive norm explanation

The constant high learning rate pushes the model farther away from `theta_0`,
whereas averaging pulls it back. Relative to the ordinary decayed-SGD endpoint,
the full SWA displacement therefore mixes opposing effects. In this experiment:

- the schedule-only displacement has cosine `-0.986` with the negative gradient
  of the checkpoint-centered quadratic;
- the averaging-only displacement has cosine `+0.999`;
- the full-SWA displacement has cosine `-0.990`.

Thus the defensible claim is about the effective regularization of the
**averaging operator**, conditional on its trajectory. It would be misleading
to report one scalar regularizer for the entire schedule-plus-average package.

This distinction is consistent with the original SWA construction, which
combines late-trajectory weight averaging with a learning-rate schedule designed
to explore a wider region of the basin:
<https://arxiv.org/abs/1803.05407>.

## Reviewer-facing interpretation

The study directly supports the "interpretability tool" claim:

1. SWA was not assigned a known analytic penalty.
2. The method compared several plausible norm and spectral families.
3. It identified a qualitatively different family: L2 distance to a historical
   checkpoint rather than L2 distance to the origin.
4. An averaging ablation verified that this family explains the causal
   displacement.
5. Explicit retraining with the estimated coefficient reproduced the SWA
   solution without averaging.

A concise rebuttal claim is:

> On a depth-3, width-256 MNIST MLP, we separate SWA's high-learning-rate
> trajectory from weight averaging using matched branches from a shared
> checkpoint. The estimator identifies a checkpoint-centered quadratic
> `lambda ||theta-theta_0||^2`, rather than ordinary weight decay, explaining
> `92.3 +/- 0.5%` of the endpoint gradient and `99.7%` of the averaging
> displacement. Replacing averaging with this fitted explicit penalty recovers
> the SWA direction (cosine `0.9993`) and all training predictions across ten
> seeds.

## Artifacts

- Experiment: `analysis/swa_effective_regularizer/run.py`
- Aggregation: `analysis/swa_effective_regularizer/analyze.py`
- Figure: `analysis/plot_swa_effective_regularizer/run.py`
- Generated summary: `data/generated/swa_effective_regularizer/summary.json`
- Figure: `results/figures/swa_effective_regularizer.pdf`
- Checkpoints: `data/generated/swa_effective_regularizer/checkpoints/`

No manuscript files were edited.
