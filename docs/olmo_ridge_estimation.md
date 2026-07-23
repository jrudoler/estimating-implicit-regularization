# OLMo Ridge Estimation

This note documents the method implemented in
[`analysis/olmo_ridge_estimation/run.py`](../analysis/olmo_ridge_estimation/run.py).
The goal is to estimate a scalar ridge penalty that is consistent with the
observed loss gradient of an OLMo checkpoint on a selected subset of its
pretraining data.

## Experiment Setup

The default experiment uses:

- Model: `allenai/OLMo-2-0425-1B`
- Revision: `stage1-step1907359-tokens4001B`
- Dataset repo: `allenai/olmo-mix-1124`
- Dataset file: `data/wiki/wiki-0001.json.gz`
- Token budget: `50,000,000` raw packed tokens
- Sequence length: `2048`
- Cache root: `$HF_HOME`, defaulting to `/shared_data0/jrudoler/.cache/huggingface`

The script intentionally downloads only the requested wiki shard with
`huggingface_hub.hf_hub_download(...)`. It does not call
`load_dataset("allenai/olmo-mix-1124")`, so it does not request the full
OLMo-mix pretraining corpus.

Text is streamed from the gzipped JSONL file, tokenized with the model tokenizer
without adding tokenizer-level special tokens, and an EOS token is appended at
document boundaries when the tokenizer has one. The token stream is packed into
fixed-length blocks. Incomplete final blocks are dropped.

## Estimation Target

The project convention for scalar ridge is

```text
R(theta) = lambda * ||theta||_2^2
```

At a fully converged regularized optimum, the first-order stationarity condition
would be

```text
grad L(theta*) + grad R(theta*) = 0.
```

Since `grad R(theta) = 2 * lambda * theta`, the scalar ridge model implies

```text
g + 2 * lambda * theta ~= 0,
```

where `g = grad L(theta)` is the gradient of mean next-token cross-entropy on
the selected data subset.

The closed-form least-squares projection onto the ridge direction is therefore

```text
lambda_hat = - <theta, g> / (2 * <theta, theta>).
```

The script also reports:

- `lambda_hat_nonnegative`: `max(lambda_hat, 0)` for a nonnegative ridge
  sensitivity check.
- `lambda_mean_loss`: the same value as `lambda_hat`, expressed relative to
  mean next-token cross-entropy.
- `lambda_sum_loss`: `lambda_mean_loss * predicted_tokens`, useful when
  comparing against a summed-token objective.
- `weight_decay_equivalent_sum_loss`: `2 * lambda_sum_loss`, the coefficient
  multiplying `theta` under an AdamW-style decay term.
- `cosine_alignment`: alignment between `-g` and `theta`.
- `residual_ratio`: relative norm of `g + 2 * lambda_hat * theta`.

## Loss Gradient

The script computes the gradient of mean next-token cross-entropy over the
packed wiki blocks. For each batch:

1. Run the causal LM forward pass.
2. Compare `logits[:, :-1, :]` against `input_ids[:, 1:]`.
3. Compute summed cross-entropy over the batch.
4. Divide that batch loss by the total number of predicted tokens in the whole
   selected subset.
5. Call `.backward()`.

After all batches, the model's `.grad` tensors contain the gradient of the mean
cross-entropy over the selected subset, not the gradient of a sum over tokens.
This normalization matters because the numerical value of `lambda_hat` depends
on the scale of the loss.

## Handling The Transformer Architecture

The transformer is handled as a differentiable black-box PyTorch module. The
script does not derive attention, MLP, layer norm, rotary embedding, or residual
stream derivatives manually. Hugging Face performs the forward pass and PyTorch
autograd backpropagates through the entire model.

Mathematically, the estimator treats selected parameter tensors as if they were
flattened and concatenated into one vector. Computationally, it does not
materialize a second billion-parameter vector. Instead, for each selected tensor
it accumulates the sufficient statistics:

```text
<theta, grad>
||theta||_2^2
||grad||_2^2
number of parameters
```

Summing these quantities over tensors is exactly equivalent to flattening and
concatenating the selected tensors for the scalar ridge projection.

Tied parameters are counted once. The implementation identifies duplicate
parameters by storage pointer, storage offset, and number of elements, which
prevents tied embedding/output-head weights from contributing twice.

## Parameter Scopes

The output reports three main scopes.

### Attention + MLP Scope

This is the primary model-grid estimate for within-family comparisons. It
includes transformer-block attention and MLP parameters, while excluding
embedding/output-head and normalization parameters. The estimate is computed by
summing `<theta, grad>`, `||theta||_2^2`, and `||grad||_2^2` over all attention
and MLP tensors, then applying the closed-form projection once. It is not an
average of the separate attention and MLP lambdas.

### Decay-Eligible Scope

This approximates the parameter set usually subject to AdamW-style weight decay:

- Include tensors with rank at least 2.
- Exclude bias parameters.
- Exclude parameters whose names look like normalization parameters, such as
  `norm`, `layernorm`, `layer_norm`, `.ln`, or `_ln`.

This scope typically includes embedding matrices and transformer attention/MLP
weight matrices, while excluding scalar/vector normalization and bias terms.

### All-Parameter Scope

This is a sensitivity estimate over every unique parameter with a gradient. It
includes matrix weights, embeddings, normalization parameters, biases, and any
other trainable tensors exposed by the model.

The script also reports coarse group summaries based on parameter names:

- `embedding_or_lm_head`
- `attention`
- `mlp`
- `norm`
- `other`

These groups are descriptive diagnostics only. The main estimator remains a
single scalar projection within each reported scope.

## Assumptions

The estimate relies on several approximations.

- The checkpoint is treated as approximately stationary for the relevant
  training objective.
- The selected wiki shard is treated as an informative proxy for the full
  pretraining distribution, even though it is only a subset of OLMo-mix.
- The regularizer family is a scalar isotropic ridge penalty within the selected
  scope. It does not fit layerwise, blockwise, diagonal, or matrix-valued ridge.
- The optimizer geometry is not modeled. In particular, Adam/AdamW moments,
  preconditioning, learning-rate schedules, clipping, and any training dynamics
  effects are not included.
- The computation uses `model.eval()`, so stochastic training-time effects are
  disabled.
- The loss is mean next-token cross-entropy on packed text. Different token
  normalization or data selection would change the numerical scale of
  `lambda_hat`.
- Parameter filtering is name- and tensor-shape-based. It is intended to match
  common decay/no-decay conventions, not to reconstruct the exact optimizer
  parameter groups used during OLMo training.

## Interpretation

If `lambda_hat` is positive and the cosine alignment is large, the observed loss
gradient on the selected subset points substantially opposite the selected
parameter vector, which is consistent with a ridge-like stationarity residual.

If `lambda_hat` is near zero, negative, or has poor alignment, that does not by
itself mean the model was not regularized. It can also indicate any of the
following:

- The selected wiki subset is not representative of the full training gradient.
- The checkpoint is not stationary for this subset.
- The effective regularization is not well described by scalar isotropic ridge.
- Optimizer-state or data-mixture effects dominate the residual.
- The relevant penalty should be layerwise, diagonal, spectral, or otherwise
  structured rather than scalar.

The first run should therefore be interpreted as a sanity-check projection of
the observed subset gradient onto a ridge direction, not as a complete recovery
of the full OLMo training objective.

## Saved Gradients

Gradient computation is the expensive part of the experiment. Runs that pass
`--save-gradients` write reusable per-parameter gradient shards alongside the
scalar estimates.

By default, a run with output

```text
data/generated/olmo_ridge_estimation/model_grid/<run_id>.json
```

stores gradients under

```text
/shared_data0/jrudoler/inductive-bias/olmo_ridge_estimation/model_grid/gradients/<run_id>/
```

This is deliberately outside `/home/jrudoler/inductive-bias` because the home
directory has a much smaller quota. Small JSON and `.pt` summary files live under
`data/generated/`, but large gradient shards should stay on `/shared_data0`.

The directory contains:

- `manifest.json`: model/data context, gradient dtype, shard list, and a mapping
  from each parameter name to its shard, shape, group, and decay eligibility.
- `gradients-00000.pt`, `gradients-00001.pt`, ...: torch files containing
  dictionaries of `{parameter_name: gradient_tensor}`.

The saved tensors are gradients of mean next-token cross-entropy under the
recorded token budget and data file. To reuse them for another candidate bias
model, load the same model checkpoint to recover `theta`, load the gradient
shards for `g`, and evaluate the new regularizer gradient against that fixed
`(theta, g)` pair. The Hugging Face model id and revision are recorded in the
manifest, so the parameter names and shapes can be checked against the loaded
checkpoint before reuse.

## Full-Wikitext Attention/MLP Bootstrap Grid

The current LLM scaling extension focuses only on attention and MLP weights.
Use `--parameter-scope attention_mlp` to freeze embeddings, normalization
parameters, and other non-block tensors before backpropagation. This mode
reports three estimates:

- `attention_plus_mlp`: the primary estimate, computed from summed attention
  and MLP sufficient statistics.
- `groups.attention`: attention weights only.
- `groups.mlp`: MLP weights only.

Use `--dataset-source wikitext --token-budget 0` to run on the full selected
Wikitext split rather than a fixed token prefix. The maintained submission
script uses `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split `train`.

Passing `--bootstrap-batches N` changes the computation from one accumulated
model gradient to per-batch sufficient statistics. For each dataloader batch,
the script backpropagates the summed next-token cross-entropy, records
`<theta, grad_sum>` for attention, MLP, and attention+MLP, and then bootstraps
`lambda_hat` by resampling batches with replacement. Each bootstrap replicate
recomputes lambda from the resampled summed numerator and token count; it does
not average per-batch lambdas.

The historical bootstrap grid covered at least three checkpoints per labeled
family, with maximum nominal model size near 15B:

- Gemma 3: 270M, 1B, 4B, 12B.
- Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B.
- Qwen3 Base: 0.6B, 4B, 14B.
- OLMo 2 stage1: 1B, 7B, 13B.

The OLMo entries are not a clean release-matched size ladder. The 1B checkpoint
is `allenai/OLMo-2-0425-1B`, while the 7B and 13B checkpoints are from
`allenai/OLMo-2-1124-*`; their pretraining token budgets also differ. Retain
those outputs as release-qualified diagnostics, not as a controlled scaling
curve.

Future controlled scaling runs use the retrained Pythia deduplicated suite:

- `EleutherAI/pythia-1b-deduped`
- `EleutherAI/pythia-2.8b-deduped`
- `EleutherAI/pythia-6.9b-deduped`
- `EleutherAI/pythia-12b-deduped`

These checkpoints belong to one scaling suite with a common dataset and
training design. The 12B endpoint is opt-in and should not be run until smaller
diagnostic models show that the ridge projection has meaningful cosine
alignment or projection R-squared.

Small JSON/PT summaries are written under
`data/generated/olmo_ridge_estimation/bootstrap_grid/`. This path intentionally
does not use `--save-gradients`, since the bootstrap summaries only need
attention/MLP sufficient statistics and confidence intervals.

## Exact Chunk-Gradient Diagnostics

Passing `--chunk-token-budget N` now uses a dedicated path that accumulates an
actual summed loss gradient within each contiguous chunk. At each chunk
boundary, the gradient is normalized by the chunk's predicted-token count and
the output retains, separately for attention, MLP, and their union:

- `<theta, g_chunk>` and `||theta||^2`
- `||g_chunk||^2`
- `lambda_hat`
- cosine alignment between `-g_chunk` and `theta`
- projection `R^2 = cosine_alignment^2`
- residual norm ratio
- chunk loss, perplexity, token count, and batch count

The previous chunk files retained only lambda samples and cannot be upgraded to
these diagnostics without rerunning the gradients. The new submission script
uses the suffix `_wikitext103_full_chunk10m_fit_v2` so it does not overwrite
those historical outputs.

The full-corpus gradient norm is still not recoverable by summing chunk gradient
norms because cross-chunk gradient inner products are not retained. Exact
full-corpus cosine therefore requires either one full accumulated gradient or
retaining enough gradient information to recover those cross terms.

Layerwise sufficient statistics are a useful possible extension, but are
deliberately deferred. The current experiment should first establish that the
coarser attention/MLP ridge direction explains a non-negligible fraction of the
chunk gradients.

## Commands

Smoke test:

```bash
HF_HOME=/shared_data0/jrudoler/.cache/huggingface \
PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/run.py \
    --token-budget 1048576 \
    --output data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_smoke_1m.json \
    --stats-output data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_smoke_1m.pt
```

Full 50M-token run:

```bash
sbatch scripts/run_olmo_ridge_estimation.sbatch
```

Submit the multi-model grid on standby H200 GPUs:

```bash
bash scripts/submit_llm_ridge_grid.sh
```

The default grid includes OLMo 2 1B at `stage1-step1907359-tokens4001B`,
OLMo 2 7B at `stage1-step928646-tokens3896B`, Qwen3 4B Base, Qwen2.5 0.5B,
and Qwen2.5 1.5B. Ai2 does not currently expose a public OLMo 2 4B language
model checkpoint; the Qwen3 4B Base run is the included 4B-scale text-model
comparison. Set `INCLUDE_GATED_GEMMA=1` to also submit the gated
`google/gemma-3-1b-pt` job if the local Hugging Face credentials have access.

Submit the controlled extension grid without overwriting the existing completed
model-grid outputs:

```bash
bash scripts/submit_llm_ridge_extension_grid.sh smoke
```

After the smoke jobs pass, submit the full 50M-token extension jobs:

```bash
bash scripts/submit_llm_ridge_extension_grid.sh full
```

The extension grid writes small JSON/PT summaries under
`data/generated/olmo_ridge_estimation/model_grid/` and large gradient shards
under `/shared_data0/jrudoler/inductive-bias/olmo_ridge_estimation/model_grid/gradients/`.
It adds `google/gemma-3-270m`, `google/gemma-3-1b-pt`, `Qwen/Qwen2.5-3B`, and
`Qwen/Qwen2.5-7B`; the Qwen2.5 7B job uses gradient checkpointing.

Submit the full-Wikitext attention/MLP bootstrap grid:

```bash
bash scripts/submit_llm_ridge_bootstrap_grid.sh smoke
```

After the smoke jobs pass, submit the full Wikitext jobs:

```bash
bash scripts/submit_llm_ridge_bootstrap_grid.sh full
```

Plot the bootstrap estimates and batch-resampling intervals:

```bash
PYTHONPATH=src uv run python analysis/plot_llm_ridge_bootstrap_grid/run.py \
    --output results/figures/llm_ridge_bootstrap_grid.pdf
```

Run a short exact-gradient smoke diagnostic on the clean Pythia ladder:

```bash
TOKEN_BUDGET=1048576 \
CHUNK_TOKEN_BUDGET=1048576 \
SUFFIX=wikitext103_1m_chunk_fit_smoke \
MODEL_SLUGS="pythia_1b_deduped pythia_28b_deduped" \
    scripts/submit_llm_ridge_chunk_grid.sh
```

After the fit-quality smoke test passes, run the sub-12B Pythia diagnostics on
full Wikitext. Add `pythia_69b_deduped` only after checking the smaller results:

```bash
MODEL_SLUGS="pythia_1b_deduped pythia_28b_deduped" \
    scripts/submit_llm_ridge_chunk_grid.sh
```

OLMo 2 remains available only as an explicit release-qualified diagnostic:

```bash
INCLUDE_OLMO_DIAGNOSTICS=1 \
MODEL_SLUGS="olmo2_1b_stage1 olmo2_7b_stage1_final" \
    scripts/submit_llm_ridge_chunk_grid.sh
```

The 12B Pythia endpoint requires `INCLUDE_LARGE_ENDPOINTS=1`. Do not enable it
until smaller models establish that the projection fit is scientifically
useful.

The first exact 1M-token float32 smoke diagnostics did not pass this gate:

| Model | Attention + MLP cosine | Projection R-squared |
| --- | ---: | ---: |
| Pythia 1B deduped | `-1.42e-05` | `2.02e-10` |
| Pythia 2.8B deduped | `-1.55e-04` | `2.41e-08` |
| OLMo 2 1B | `3.71e-05` | `1.37e-09` |
| OLMo 2 7B | `-7.76e-05` | `6.02e-09` |

All residual ratios were effectively one. Consequently, no full-Wikitext,
Pythia 6.9B, or 12B-14B fit-diagnostic jobs were submitted. The chunk script
requires an explicit `MODEL_SLUGS` list to prevent accidental broad reruns.

Chunk jobs write JSON/PT summaries under
`data/generated/olmo_ridge_estimation/chunk_grid/`. For displaying heterogeneity
among 10M-token subsets, prefer chunk standard deviation over SEM. SEM is only
appropriate when the estimand is the precision of the mean across chunks.

The historical lambda-only chunk figure is generated with:

```bash
PYTHONPATH=src uv run python analysis/plot_llm_ridge_bootstrap_grid/run.py \
    --input-dir data/generated/olmo_ridge_estimation/chunk_grid \
    --pattern '*_wikitext103_full_chunk10m.json' \
    --output results/figures/llm_ridge_chunk_grid.pdf
```

Summarize completed full-grid outputs ordered by transformer-block parameter
count:

```bash
PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/summarize_model_grid.py \
    --output data/generated/olmo_ridge_estimation/model_grid/summary.md
```

Snakemake target:

```bash
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm \
    data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_50m.json
```
