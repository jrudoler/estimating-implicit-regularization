# Training and Data Loading Profiling Tools

This directory contains tools to help diagnose low GPU utilization and identify bottlenecks in model training and data loading.

## Quick Start (Slurm Cluster)

**Recommended: Interactive session for batch size and GPU memory testing**
```bash
# 1. Allocate resources (you'll still be on login node)
salloc --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:30:00

# 2. Use srun to run commands on the allocated compute node
srun uv run python tests/profile_lightning.py --batch-sizes 128 256 512 1024 --num-workers-list 4 8

# View results table and check GPU memory utilization
# Exit when done
exit
```

**Alternative: SSH to allocated node**
```bash
# After salloc, check allocated node
squeue -u $USER

# SSH to the node (e.g., dgx-b200-01)
ssh dgx-b200-01

# Now run commands directly (no srun needed)
uv run python tests/profile_lightning.py --batch-sizes 128 256 512 1024
```

**Alternative: Profile data loading only**
```bash
# In salloc session, profile data loading
uv run python tests/profile_training.py --profile-dataloader-only --num-workers-list 0 2 4 8 16
```

**Alternative: Submit batch job**
```bash
# Quick data loading profiling (~15 min)
sbatch tests/profile_dataloader_only.slurm

# Full profiling (~30 min)
sbatch tests/profile_training.slurm

# Check results
tail -f logs/slurm/profile_*.log
```

## Tools

### 1. `profile_lightning.py` - Lightning Profiler (Recommended)

Uses PyTorch Lightning's built-in PyTorch Profiler to compare different Trainer and DataLoader configurations. **Best for testing batch sizes and GPU memory utilization.**

**Features:**
- Uses Lightning's PyTorchProfiler (integrated with PyTorch Profiler)
- Compares multiple batch sizes and num_workers configurations
- Tracks GPU memory utilization per configuration
- Generates TensorBoard traces for detailed analysis
- Saves comparison results to JSON

**Usage:**

```bash
# In salloc session, test different batch sizes:
uv run python tests/profile_lightning.py --batch-sizes 128 256 512 1024

# Test different num_workers:
uv run python tests/profile_lightning.py --num-workers-list 0 2 4 8

# Test both batch sizes and num_workers:
uv run python tests/profile_lightning.py \
    --batch-sizes 256 512 1024 \
    --num-workers-list 4 8

# Compare GPU vs CPU (recommended by admins):
uv run python tests/profile_lightning.py \
    --batch-sizes 256 512 \
    --num-workers-list 4 8 \
    --compare-cpu

# Sweep CPU DDP process counts on CPU:
uv run python tests/profile_lightning.py \
    --device cpu \
    --cpu-devices-list 1 2 4 8 \
    --batch-sizes 256 \
    --num-workers-list 4

# Test only on CPU:
uv run python tests/profile_lightning.py \
    --batch-sizes 256 512 \
    --device cpu

# Custom model size:
uv run python tests/profile_lightning.py \
    --batch-sizes 256 512 \
    --depth 3 \
    --width 512
```

**Output:**
- Comparison table showing training speed and GPU memory usage per configuration
- **CPU vs GPU comparison** (when using `--compare-cpu`) showing speedup/slowdown
- Profiler traces in TensorBoard format (view with `tensorboard --logdir logs/profiling`)
- CSV metrics for each configuration
- JSON summary with all results
- Recommendations on whether to use GPU or CPU based on performance

**View Results:**
```bash
# View TensorBoard traces
tensorboard --logdir logs/profiling

# Check comparison results
cat logs/profiling/comparison_results.json
```

### 2. `profile_training.py` - Comprehensive Training Profiler

Profiles data loading and training performance to identify bottlenecks.

**Features:**
- Profiles data loading with different `num_workers` values
- Profiles training step performance (forward, backward, optimizer step)
- Profiles end-to-end training (data loading + compute)
- Monitors GPU memory usage
- Provides recommendations based on profiling results

**Usage:**

```bash
# Basic usage
uv run python tests/profile_training.py

# Custom configuration
uv run python tests/profile_training.py \
    --dataset mnist \
    --batch-size 256 \
    --depth 3 \
    --width 256 \
    --num-workers-list 0 2 4 8 16 \
    --num-batches 50

# Only profile data loading (faster)
uv run python tests/profile_training.py --profile-dataloader-only
```

**Output:**
- Data loading performance for each `num_workers` value
- Training step breakdown (forward/backward/step times)
- End-to-end performance with data/compute fractions
- GPU statistics
- Recommendations for optimization

**Example Output:**
```
Data Loading Performance
============================================================
  num_workers=0 mean_load_time              : 0.012345
  num_workers=2 mean_load_time                : 0.006789
  num_workers=4 mean_load_time                 : 0.005432
  num_workers=8 mean_load_time                : 0.005123
  Best num_workers: 8

Bottleneck Analysis
  Data loading fraction: 65.23%
  ⚠️  WARNING: Data loading is the bottleneck!
  - Consider increasing num_workers
  - Consider using pin_memory=True (if using GPU)
```

### 2. `monitor_gpu.py` - Real-time GPU Monitoring

Monitors GPU utilization and memory usage in real-time during training.

**Usage:**

```bash
# Monitor GPU in real-time (until Ctrl+C)
uv run python tests/monitor_gpu.py

# Monitor for 60 seconds with 1-second intervals
uv run python tests/monitor_gpu.py --interval 1 --duration 60

# Save statistics to CSV file
uv run python tests/monitor_gpu.py --interval 1 --duration 300 --output gpu_stats.csv
```

**Output:**
- Real-time GPU utilization percentage
- Memory utilization percentage
- GPU temperature
- Summary statistics (average, max utilization)

## Common Issues and Solutions

### Low GPU Utilization (< 10%)

**Symptoms:**
- GPU utilization consistently below 10%
- GPU memory usage very low (< 5%)
- CPU utilization around 50-60%

**Likely Causes:**
1. **Data loading bottleneck** - GPU is waiting for data
   - **Solution:** Increase `num_workers` in DataLoader
   - **Solution:** Use `pin_memory=True` for GPU training
   - **Solution:** Use `persistent_workers=True` to avoid worker restart overhead

2. **Model too small** - Not enough work for GPU
   - **Solution:** Increase batch size
   - **Solution:** Consider using CPU instead of GPU
   - **Solution:** Use `dgx-b200-mig45` partition (cheaper, 45GB RAM)

3. **Inefficient data preprocessing**
   - **Solution:** Preprocess data offline
   - **Solution:** Use faster transforms
   - **Solution:** Cache preprocessed data

### Low GPU Memory Usage (< 5%)

**Symptoms:**
- GPU memory usage very low
- Model fits easily in memory

**Solutions:**
- Use `dgx-b200-mig45` partition instead of full GPU
- Consider using CPU with more cores
- Increase batch size to better utilize GPU

### High Data Loading Time

**Symptoms:**
- Data loading takes > 50% of total time
- Training step is fast but overall throughput is low

**Solutions:**
1. **Increase num_workers:**
   ```python
   DataLoader(..., num_workers=8, pin_memory=True, persistent_workers=True)
   ```

2. **Use pin_memory for GPU:**
   ```python
   DataLoader(..., pin_memory=True)  # Only if using GPU
   ```

3. **Preprocess data offline:**
   - Save preprocessed data to disk
   - Load from disk instead of processing on-the-fly

## Running on Slurm Cluster

### Option 1: Interactive Session with `salloc` (Recommended for Iterative Profiling)

**Important:** After `salloc` allocates resources, you need to run commands on the allocated compute node. You have two options:

#### Option 1a: Use `srun` within `salloc` session (Recommended)

```bash
# Request resources interactively (you'll still be on login node)
salloc --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:30:00

# Note: You're still on the login node, but resources are allocated
# Use srun to run commands on the allocated compute node:

# Quick data loading profiling
srun uv run python tests/profile_training.py --profile-dataloader-only --num-workers-list 0 2 4 8 16

# Full profiling
srun uv run python tests/profile_training.py --num-batches 30 --num-workers-list 0 2 4 8

# Test different batch sizes with Lightning profiler
srun uv run python tests/profile_lightning.py --batch-sizes 128 256 512 1024 --num-workers-list 4 8

# When done, exit the allocation
exit
```

#### Option 1b: SSH to the allocated node

```bash
# Request resources and get node name
salloc --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:30:00

# Check which node was allocated (from salloc output or use squeue)
squeue -u $USER
# Look for the node name (e.g., dgx-b200-01)

# SSH to that node (if you have access)
ssh dgx-b200-01

# Now you're on the compute node, run commands directly:
uv run python tests/profile_lightning.py --batch-sizes 128 256 512 1024

# When done, exit SSH and then exit salloc
exit  # exits SSH
exit  # exits salloc
```

**Advantages:**
- Interactive - see results immediately
- Can iterate quickly - adjust parameters and rerun
- Good for experimentation

### Option 2: Batch Job with `sbatch` (Recommended for One-Time Profiling)

Submit a profiling job:

```bash
# Profile data loading only (faster, ~15 min)
sbatch tests/profile_dataloader_only.slurm

# Profile full training (slower, ~30 min)
sbatch tests/profile_training.slurm

# Check job status
squeue -u $USER

# View results
tail -f logs/slurm/profile_*.log
```

**Advantages:**
- Non-interactive - submit and forget
- Results saved to log files
- Good for one-time profiling

### Option 3: Direct `srun` (Quick One-Off Commands)

Run a single command directly:

```bash
# Profile data loading
srun --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:15:00 \
    uv run python tests/profile_training.py --profile-dataloader-only

# Profile full training
srun --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:30:00 \
    uv run python tests/profile_training.py --num-batches 30
```

## Benchmarking Workflow

1. **Start with interactive session:**
   ```bash
   salloc --partition=dgx-b200-mig45 --gres=gpu:1 --cpus-per-task=8 --mem=45G --time=00:30:00
   ```

2. **Profile with Lightning profiler (compare batch sizes and CPU vs GPU):**
   ```bash
   # Use srun to run on allocated compute node
   # Compare GPU vs CPU (as admins suggested)
   srun uv run python tests/profile_lightning.py \
       --batch-sizes 128 256 512 \
       --num-workers-list 4 8 \
       --compare-cpu
   ```

3. **Profile data loading first (fastest):**
   ```bash
   srun uv run python tests/profile_training.py --profile-dataloader-only \
       --num-workers-list 0 2 4 8 16
   ```

4. **Profile full training with optimal num_workers:**
   ```bash
   srun uv run python tests/profile_training.py \
       --num-batches 30 \
       --num-workers-list 4 8  # Use best from step 3
   ```

5. **View profiler traces:**
   ```bash
   # In another terminal (or after exiting salloc), view TensorBoard
   tensorboard --logdir logs/profiling
   ```

6. **Monitor during actual training:**
   ```bash
   # In salloc session, start training in background (use srun)
   srun uv run python experiments/l2_train_and_recover.py ... &
   
   # Monitor GPU in same session
   srun uv run python tests/monitor_gpu.py --interval 1 --duration 300
   ```

7. **Compare CPU vs GPU:**
   ```bash
   # Profile with GPU (in salloc session, use srun)
   srun uv run python tests/profile_training.py --device cuda
   
   # Profile with CPU (can run on login node or request CPU partition)
   uv run python tests/profile_training.py --device cpu
   ```

## Partition Selection

Based on your GPU memory usage (< 5%), use the **`dgx-b200-mig45`** partition:
- Cheaper than full `dgx-b200` partition
- 45GB RAM (sufficient for your models)
- Better resource utilization

If you need more RAM, use `dgx-b200` instead.

## Integration with Training Jobs

You can add profiling to your training Slurm scripts:

```bash
#!/bin/bash
#SBATCH --partition=dgx-b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G

# Quick profile before training (optional)
uv run python tests/profile_training.py \
    --profile-dataloader-only \
    --num-workers-list 0 2 4 8 \
    > logs/profile_${SLURM_JOB_ID}.log 2>&1

# Then run actual training
uv run python experiments/l2_train_and_recover.py ...
```

## Tips

1. **Start with data loading profiling** - This is often the bottleneck
2. **Test different num_workers** - Optimal value depends on your system
3. **Monitor during actual training** - Use `monitor_gpu.py` to see real utilization
4. **Compare configurations** - Profile with different batch sizes, num_workers, etc.
5. **Check jobstats** - Use `jobstats JOBID` on completed jobs to see overall utilization

## References

- [PyTorch DataLoader Performance Tuning](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)
- [Lightning DataLoader Best Practices](https://lightning.ai/docs/pytorch/stable/guides/data.html)

