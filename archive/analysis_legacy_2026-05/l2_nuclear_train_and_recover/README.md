# L2 + Nuclear Norm Identifiability Experiment

This experiment tests whether L2 and nuclear norm regularization penalties are identifiable when applied jointly to deep ReLU networks. The goal is to determine if these two types of regularization are too correlated to be recovered independently, or if they can be successfully disentangled.

## Experiment Design

1. **Training Phase**: Train a deep ReLU classifier with BOTH L2 and nuclear norm penalties at known ground-truth values
2. **Recovery Phase**: Use gradient matching with a `JointBias` model (combining `RidgeBias` + `NuclearNormBias`) to estimate both penalty coefficients
3. **Analysis**: Compare estimated vs true parameters to assess identifiability

## Files

- `sweeps/l2_nuclear_recovery.yaml` - W&B sweep configuration
- `experiments/l2_nuclear_train_and_recover.py` - Training and recovery script
- `scripts/wandb_sweep.slurm` - Canonical Slurm W&B agent launcher

## Running the Experiment

### 1. Register the sweep

```bash
cd /path/to/inductive-bias
wandb sweep sweeps/l2_nuclear_recovery.yaml
```

This will output a sweep ID like `your-entity/your-project/abc123xyz`.

### 2. Launch sweep agents

**On a GPU cluster (recommended):**

```bash
sbatch scripts/wandb_sweep.slurm <SWEEP_ID>
```

You can launch multiple jobs to parallelize:

```bash
for i in {1..10}; do
  sbatch scripts/wandb_sweep.slurm <SWEEP_ID>
done
```

**Locally (for testing):**

```bash
uv run wandb agent <SWEEP_ID>
```

### 3. Monitor progress

View the sweep dashboard at:
```
https://wandb.ai/<entity>/<project>/sweeps/<SWEEP_ID>
```

### 4. Analyze results

Once the sweep completes, inspect metrics directly in W&B:

```bash
https://wandb.ai/<entity>/<project>/sweeps/<SWEEP_ID>
```

## Key Metrics

The experiment logs several metrics to assess identifiability:

- **`l2_rel_error`**: Relative error for L2 parameter recovery
- **`nuclear_rel_error`**: Relative error for nuclear norm parameter recovery
- **`mean_rel_error`**: Average of both relative errors
- **`max_rel_error`**: Maximum of both relative errors (worst-case identifiability)

**Good identifiability** is indicated by:
- Low relative errors (< 0.1 or 10%)
- Consistent recovery across different random seeds
- No systematic bias in over/under-estimation

**Poor identifiability** is indicated by:
- High relative errors (> 0.5 or 50%)
- High variance across seeds
- Strong negative correlation in estimation errors (suggesting the two penalties trade off)

## Sweep Configuration

The sweep explores:
- **L2 penalties**: [0.001, 0.01, 0.05]
- **Nuclear norm penalties**: [0.001, 0.01, 0.05]
- **Network depths**: [2, 3]
- **Network widths**: [128, 256]
- **Random seeds**: [42, 123, 456, 789, 999]

Total configurations: 3 × 3 × 2 × 2 × 5 = **180 runs**

## Expected Outcomes

**Hypothesis 1: Identifiable**
- If L2 and nuclear norm are sufficiently different, both parameters should be recoverable with low error
- Errors should be independent (low correlation)

**Hypothesis 2: Confounded**
- If the two penalties have similar effects on the gradient, they may trade off
- High errors with negative correlation (overestimating one compensates for underestimating the other)
- May depend on the ratio of penalties and network architecture

## Troubleshooting

**Sweep runs fail immediately:**
- Check that `data/` directory exists and contains MNIST
- Verify GPU availability if using CUDA

**High training loss:**
- May indicate penalties are too large; check hyperparameters
- Try reducing learning rate

**Recovery fails to converge:**
- Increase `bias_max_epochs` or `bias_patience`
- Try different `bias_lr` values
- Check that predictive model trained successfully

**Out of memory:**
- Reduce `batch_size`
- Reduce `width` or `depth`
- Use gradient checkpointing (not currently implemented)
