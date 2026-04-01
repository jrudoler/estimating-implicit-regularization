# Paper Review and Next Steps

Date: 2026-03-31

## Paper Status

The paper has a solid theoretical framework (gradient-matching to estimate implicit regularization) with clean proofs in the appendix. Current coverage:

1. **Complete:** Framework setup, stationarity/gradient-matching derivation, elastic net recovery, GD-on-OLS early stopping, appendix uniqueness proofs
2. **Skeletal/placeholder:** Dropout (figure exists, text is one paragraph), "Neural nets stuff" section, "Identifying regularization in complex models" (just notes), model evaluation section (marked red with TODOs)
3. **Missing entirely:** All nonlinear multi-geometry results from the last month of experiments

## Key Gaps Between Paper and Experiments

~4 weeks of nonlinear work (bootstrap resampling, stacked shared-lambda, power-family geometry, multi-component recovery, replicate ablation) has produced substantial results that do not appear in the paper. Most paper-ready findings:

- **Multi-geometry recovery**: cosine >0.93 for single-L2, >0.92 for two-component mixtures — the headline nonlinear result
- **Exact sanity checks**: cosine 1.0 across all cases — proves the estimator is correct; bottleneck is approximate stationarity, not the fitting routine
- **Replicate ablation**: more bootstrap retrains do not help — the problem is model mismatch, not sample size; a principled scope limitation

## Next Steps

### 1. Fill in the missing paper sections (highest priority)

- **Section 4.4 ("Neural nets stuff"):** Write up the nonlinear multi-geometry suite. The `single_l2` case (cosine 0.94) is the cleanest demonstration; then show multi-component as a harder test. Include the exact sanity sweep as evidence that the estimator itself works.
- **Section 3.4 (model evaluation, currently red):** Formalize the cosine similarity and residual norm metrics already used in experiments. The gradient-space regression interpretation is exactly what the experiments do — connect the text to the actual evaluation pipeline.
- **Dropout section:** Either flesh it out with proper experimental details or cut it and focus on the stronger nonlinear results.

### 2. Frame the identifiability story

The experiments reveal a core tension: aggregate gradient geometry is recoverable, but individual components in a mixture are not always separable. Frame this as a strength:
- Exact sanity sweep → positive identifiability result (the estimator is faithful)
- Multi-component exchangeability (L1/L2 within elementwise family) → a **diagnostic**, revealing the actual geometry of the regularization landscape
- Replicate ablation negative result → supports approximate stationarity as the bottleneck, not insufficient data; include as principled scope limitation

### 3. Strengthen the experiments section structure

Current paper jumps from linear OLS to "neural nets stuff." Suggested flow:
1. Elastic net recovery (existing) — proves the method works with known ground truth
2. GD on OLS early stopping (existing) — validates against theory
3. **New:** Nonlinear single-component recovery (L2, L1, nuclear) in deep ReLU nets — extends to the practical regime
4. **New:** Multi-component geometry recovery — shows the method handles mixtures and diagnoses identifiability
5. Dropout (if retained) — application to a real implicit bias

### 4. Minor paper fixes

- **Line 53:** TODO "something something deep learning yay" needs a real bullet — likely summarizing the nonlinear recovery results
- **Lines 414-448:** Red-text model evaluation section needs to be finalized
- **`\jhr{}` notes:** ~8 scattered throughout; resolve or remove before submission
- **Figure filename:** `figures/sweep_krcszr6z_...` (W&B artifact name) — rename for readability

### 5. Experimental directions worth pursuing

- **Stationarity gap analysis:** The docs consistently identify approximate stationarity as the bottleneck. A controlled experiment measuring stationarity residual vs. recovery error across training durations would quantify this and give practitioners guidance on when the method is reliable.
- **Scale up architectures:** Current experiments use depth-2, width-32 nets on 12-dim input. A modest scale-up (e.g., small ConvNet, wider MLP) would strengthen the "practical" claim. Maps to the Konrad note in the paper about fitting big models like ViT or small LLMs.
- **Retrain validation (mode 2):** Section 3.3 defines retrain-with-estimated-regularizer as a validation mode, but the nonlinear experiments don't do this. Showing that a network retrained with the estimated BER produces similar predictions would be a compelling end-to-end validation.

### 6. Framing / positioning

The abstract promises "characterize implicit regularization of training techniques like dropout and early stopping in practical settings." The strongest support for this claim is the multi-geometry suite, not dropout. Consider reframing contributions around:

- A general-purpose estimation framework (theory)
- Exact recovery in linear settings (validation)
- Geometry recovery in nonlinear settings with principled identifiability diagnostics (main result)
- Concrete characterization of when/why accuracy degrades (honest scope)
