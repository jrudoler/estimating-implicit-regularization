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

The output reports two main scopes.

### Decay-Eligible Scope

This is the primary estimate. It approximates the parameter set usually subject
to AdamW-style weight decay:

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
data/generated/olmo_ridge_estimation/model_grid/gradients/<run_id>/
```

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

The default grid includes OLMo 2 1B, OLMo 2 7B, Qwen3 4B Base, Qwen2.5 0.5B,
and Qwen2.5 1.5B. Ai2 does not currently expose a public OLMo 2 4B language
model checkpoint; the Qwen3 4B Base run is the included 4B-scale text-model
comparison. Set `INCLUDE_GATED_GEMMA=1` to also submit the gated
`google/gemma-3-1b-pt` job if the local Hugging Face credentials have access.

Snakemake target:

```bash
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm \
    data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_50m.json
```
