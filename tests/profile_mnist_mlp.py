#!/usr/bin/env python3
"""
mnist_sweep.py — sweep small MLPs for MNIST with PyTorch Lightning.

- Accepts a YAML config via --config or uses DEFAULT_SWEEP below.
- Saves a CSV of throughput results to --out.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import argparse
import csv
import itertools
import os
import sys
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import CSVLogger
import statistics
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

# Optional YAML support
try:
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# ---- Defaults for the sweep space (edit here or supply a YAML) ----
DEFAULT_SWEEP: Dict[str, Sequence[Any]] = {
    # Core sweep knobs (lists or scalars)
    "batch_size": [2048, 8192, 16384, 32768],
    "gpu_resident": [True, False],
    "precision": ["bf16-mixed", "32-true"],  # will auto-fallback to 32 on CPU
    "width": [128, 256, 512],
    "depth": [5],
    "compile_model": [False, True],
    "num_workers": [0, 4, 8],  # 0 when gpu_resident=True is fine
    # Optimizer hyperparams
    "optimizer": ["sgd"],  # or "adamw"
    "base_lr": [1e-2],
    "weight_decay": [0.0],
    # Measurement controls (scalars)
    "warmup_steps": [20],
    "measure_steps": [200],
}

# ---- Low-level perf toggles ----
torch.backends.cuda.matmul.allow_tf32 = True  # NVIDIA TF32
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# ---- Callback ----
class GPUStatsSampler(pl.Callback):
    """Sample GPU utilization and memory with minimal overhead."""

    def __init__(self, warmup_steps: int, measure_steps: int) -> None:
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self._count = 0
        self._dev_index: int = 0
        self.util_samples: list[int] = []
        self.mem_used_bytes: list[int] = []

        # NVML fallback state
        self._use_torch_util = hasattr(torch.cuda, "utilization")
        self._use_device_mem_used = hasattr(torch.cuda, "device_memory_used")
        self._pynvml = None
        self._nvml_handle = None

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        if stage != "fit" or not torch.cuda.is_available():
            return
        dev = trainer.strategy.root_device
        self._dev_index = int(dev.index or 0)

        # Reset PyTorch allocator peak stats before measuring
        try:
            torch.cuda.reset_peak_memory_stats(self._dev_index)  # legacy alias
        except Exception:
            try:
                torch.cuda.memory.reset_peak_memory_stats(self._dev_index)  # new path
            except Exception:
                pass

        # Prepare NVML if torch.cuda.utilization is unavailable
        if not self._use_torch_util:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                self._pynvml = pynvml
                # Map local CUDA index -> NVML index respecting CUDA_VISIBLE_DEVICES
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                nvml_index = (
                    self._dev_index
                    if not visible
                    else int(visible.split(",")[self._dev_index])
                )
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            except Exception:
                self._pynvml = None
                self._nvml_handle = None

    def _sample_once(self) -> None:
        if not torch.cuda.is_available():
            return

        # Utilization (percent)
        util: int | None = None
        try:
            if self._use_torch_util:
                util = int(
                    torch.cuda.utilization(self._dev_index)
                )  # NVML-backed in PyTorch 2.9+
            elif self._pynvml and self._nvml_handle is not None:
                util = int(
                    self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                )
        except Exception:
            util = None  # ignore sampling errors

        # Global device memory used (bytes)
        mem_used: int | None = None
        try:
            if self._use_device_mem_used:
                mem_used = int(torch.cuda.device_memory_used(self._dev_index))
            else:
                free_b, total_b = torch.cuda.memory.mem_get_info(self._dev_index)
                mem_used = int(total_b - free_b)
        except Exception:
            mem_used = None

        # Record within measurement window only
        if (
            self._count > self.warmup_steps
            and self._count <= self.warmup_steps + self.measure_steps
        ):
            if util is not None:
                self.util_samples.append(util)
            if mem_used is not None:
                self.mem_used_bytes.append(mem_used)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._count += 1
        self._sample_once()

    # Convenience accessors after fit:
    def summary(self) -> dict[str, float | None]:
        to_gib = lambda b: (float(b) / (1024**3)) if b is not None else None
        avg_util = statistics.mean(self.util_samples) if self.util_samples else None
        max_util = max(self.util_samples) if self.util_samples else None
        avg_mem = statistics.mean(self.mem_used_bytes) if self.mem_used_bytes else None
        max_mem = max(self.mem_used_bytes) if self.mem_used_bytes else None
        # PyTorch allocator peaks (bytes) – only counts tensors managed by the allocator
        try:
            peak_alloc = torch.cuda.max_memory_allocated(self._dev_index)
            peak_res = torch.cuda.max_memory_reserved(self._dev_index)
        except Exception:
            peak_alloc = peak_res = 0
        return {
            "gpu_util_avg_pct": float(avg_util) if avg_util is not None else None,
            "gpu_util_max_pct": float(max_util) if max_util is not None else None,
            "gpu_mem_used_avg_GiB": to_gib(avg_mem),
            "gpu_mem_used_max_GiB": to_gib(max_mem),
            "alloc_peak_GiB": to_gib(peak_alloc),
            "reserved_peak_GiB": to_gib(peak_res),
        }


# ------------------------- Model -------------------------


class MLP(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 28 * 28,
        width: int = 256,
        depth: int = 5,
        num_classes: int = 10,
        base_lr: float = 1e-2,
        weight_decay: float = 0.0,
        optimizer_name: Literal["sgd", "adamw"] = "sgd",
        compile_model: bool = False,
        global_batch_size: int = 128,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(last, width), nn.ReLU(inplace=True)]
            last = width
        layers += [nn.Linear(last, num_classes)]
        net = nn.Sequential(*layers)
        if compile_model and hasattr(torch, "compile"):
            net = torch.compile(net, fullgraph=True)  # type: ignore[attr-defined]
        self.net = net
        self.loss = nn.CrossEntropyLoss()

        # store for optimizer config
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.global_batch_size = int(global_batch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _: int) -> None:
        x, y = batch
        pred = self(x).argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Linear LR scaling wrt global batch size (ref=128)
        lr = self.base_lr * (self.global_batch_size / 128.0)
        if self.optimizer_name == "sgd":
            opt = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        # Short warmup helps with very large batches
        sched = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# ------------------------- Data -------------------------


@dataclass
class DataConf:
    batch_size: int
    num_workers: int
    train_on_gpu: bool  # whether the Trainer uses GPU
    host_on_gpu: bool  # whether dataset tensors live on GPU
    dtype: torch.dtype  # storage dtype for features
    val_fraction: float = 0.1


class MNISTGPUDataModule(pl.LightningDataModule):
    def __init__(self, conf: DataConf) -> None:
        super().__init__()
        self.conf = conf
        self.train_ds: Optional[TensorDataset] = None
        self.val_ds: Optional[TensorDataset] = None

    def prepare_data(self) -> None:
        MNIST(root=".", train=True, download=True)
        MNIST(root=".", train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        raw = MNIST(root=".", train=True, download=False)
        x_u8 = torch.from_numpy(raw.data.numpy())  # [N, 28, 28], uint8
        y = torch.tensor(raw.targets.numpy(), dtype=torch.long)

        # Flatten
        x = x_u8.view(x_u8.size(0), -1)

        # Choose device for hosting dataset
        device = (
            torch.device("cuda", 0)
            if (self.conf.host_on_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        # Normalize on target device, store as requested dtype
        x = (
            x.to(device=device, non_blocking=self.conf.train_on_gpu)
            .to(torch.float32)
            .div_(255.0)
        )
        x.sub_(0.1307).div_(0.3081)
        x = x.to(self.conf.dtype)
        y = y.to(device=device, non_blocking=self.conf.train_on_gpu)

        # Split
        n_val = int(len(x) * self.conf.val_fraction)
        n_train = len(x) - n_val
        x_train, x_val = torch.split(x, [n_train, n_val])
        y_train, y_val = torch.split(y, [n_train, n_val])
        self.train_ds = TensorDataset(x_train, y_train)
        self.val_ds = TensorDataset(x_val, y_val)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        assert self.train_ds is not None
        # Pin memory only if data is on CPU and training on GPU
        pin = (not self.conf.host_on_gpu) and self.conf.train_on_gpu
        kwargs: Dict[str, Any] = dict(
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=pin,
            persistent_workers=self.conf.num_workers > 0,
        )
        if self.conf.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=max(2048, self.conf.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


# ------------------------- Benchmarking -------------------------


class ThroughputCallback(pl.Callback):
    def __init__(self, warmup_steps: int = 20, measure_steps: int = 200) -> None:
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self._count = 0
        self._start: Optional[float] = None
        self.samples_per_sec: Optional[float] = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        bsz = int(batch[0].size(0))
        self._count += 1
        # Start timing after warmup
        if self._count == self.warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start = time.perf_counter()
        elif self._count == self.warmup_steps + self.measure_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(
                1e-9, (time.perf_counter() - (self._start or time.perf_counter()))
            )
            seen = self.measure_steps * bsz
            self.samples_per_sec = float(seen) / float(elapsed)
            trainer.should_stop = True


# ------------------------- Utilities -------------------------


def _effective_precision(requested: str, using_gpu: bool) -> str:
    """Force 32-bit on CPU; otherwise return requested."""
    if not using_gpu:
        return "32-true"
    return requested


def _input_dtype_for_precision(precision: str) -> torch.dtype:
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    return torch.float32


def _expand_grid(space: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    # Coerce scalars to lists
    keys = list(space.keys())
    values: List[Sequence[Any]] = [
        v if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else [v]
        for v in space.values()
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _load_sweep(path: Optional[str]) -> Dict[str, Sequence[Any]]:
    if path is None:
        return DEFAULT_SWEEP
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML not installed but --config was provided.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML must define a top-level mapping.")
    # Allow either full file as the mapping or under key 'sweep'
    sweep = data.get("sweep", data) if isinstance(data, dict) else data
    if not isinstance(sweep, dict):
        raise ValueError("Invalid sweep mapping in YAML.")
    return sweep  # type: ignore[return-value]


# ------------------------- Single-run driver -------------------------


def run_once(cfg: Dict[str, Any], use_gpu: bool) -> Dict[str, Any]:
    """Execute one training-throughput measurement for a given config."""
    # Extract knobs with defaults
    batch_size: int = int(cfg.get("batch_size", 16384))
    gpu_resident: bool = bool(cfg.get("gpu_resident", True))
    precision_req: str = str(cfg.get("precision", "bf16-mixed"))
    width: int = int(cfg.get("width", 256))
    depth: int = int(cfg.get("depth", 5))
    compile_model: bool = bool(cfg.get("compile_model", True))
    num_workers: int = int(cfg.get("num_workers", 0))
    optimizer: str = str(cfg.get("optimizer", "sgd"))
    base_lr: float = float(cfg.get("base_lr", 1e-2))
    weight_decay: float = float(cfg.get("weight_decay", 0.0))
    warmup_steps: int = int(cfg.get("warmup_steps", 20))
    measure_steps: int = int(cfg.get("measure_steps", 200))

    precision = _effective_precision(precision_req, use_gpu)
    input_dtype = _input_dtype_for_precision(precision)

    dataconf = DataConf(
        batch_size=batch_size,
        num_workers=num_workers
        if not gpu_resident
        else 0,  # workers not needed if hosting on GPU
        train_on_gpu=use_gpu,
        host_on_gpu=(use_gpu and gpu_resident),
        dtype=input_dtype,
    )
    dm = MNISTGPUDataModule(dataconf)

    model = MLP(
        width=width,
        depth=depth,
        base_lr=base_lr,
        weight_decay=weight_decay,
        optimizer_name=("sgd" if optimizer.lower() == "sgd" else "adamw"),
        compile_model=(compile_model and hasattr(torch, "compile")),
        global_batch_size=batch_size,
    )

    cb = ThroughputCallback(warmup_steps=warmup_steps, measure_steps=measure_steps)
    devmon = DeviceStatsMonitor() if use_gpu else None
    gpumon = (
        GPUStatsSampler(warmup_steps=warmup_steps, measure_steps=measure_steps)
        if use_gpu
        else None
    )

    # DeviceStatsMonitor requires a logger
    logger = CSVLogger(save_dir="pl_logs", name="sweep") if use_gpu else False

    callbacks = [cb]
    if devmon:
        callbacks.append(devmon)
    if gpumon:
        callbacks.append(gpumon)

    trainer = pl.Trainer(
        accelerator=("gpu" if use_gpu else "cpu"),
        devices=1,
        precision=precision,
        max_epochs=10_000,
        logger=logger,  # <— logger ON so DeviceStatsMonitor works
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        benchmark=True,
        max_steps=10_000,
        callbacks=callbacks,
    )

    started = time.perf_counter()
    err: Optional[str] = None
    try:
        trainer.fit(model, dm)
    except Exception as e:
        err = repr(e)
    elapsed = time.perf_counter() - started

    gpu_name = (
        torch.cuda.get_device_name(0)
        if (use_gpu and torch.cuda.is_available())
        else "CPU"
    )

    # GPU stats summary
    stats = gpumon.summary() if gpumon else {}
    result: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": ("gpu" if use_gpu else "cpu"),
        "gpu_name": gpu_name,
        "batch_size": batch_size,
        "gpu_resident": gpu_resident,
        "precision_requested": precision_req,
        "precision_effective": precision,
        "width": width,
        "depth": depth,
        "compile_model": compile_model and hasattr(torch, "compile"),
        "num_workers": dataconf.num_workers,
        "optimizer": optimizer,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "samples_per_sec": float(cb.samples_per_sec or 0.0),
        "elapsed_s": float(elapsed),
        # New GPU stats:
        "gpu_util_avg_pct": stats.get("gpu_util_avg_pct"),
        "gpu_util_max_pct": stats.get("gpu_util_max_pct"),
        "gpu_mem_used_avg_GiB": stats.get("gpu_mem_used_avg_GiB"),
        "gpu_mem_used_max_GiB": stats.get("gpu_mem_used_max_GiB"),
        "alloc_peak_GiB": stats.get("alloc_peak_GiB"),
        "reserved_peak_GiB": stats.get("reserved_peak_GiB"),
        "success": err is None,
        "error": ("" if err is None else err),
        "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 6),
    }
    return result


# ------------------------- Main -------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML sweep config path (optional). If omitted, uses DEFAULT_SWEEP in the script.",
    )
    parser.add_argument(
        "--out", type=str, default="sweep_results.csv", help="CSV output path."
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Force CPU/GPU or auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of sweep combinations.",
    )
    args = parser.parse_args()

    if args.accelerator == "gpu":
        use_gpu = torch.cuda.is_available()
    elif args.accelerator == "cpu":
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    sweep_space = _load_sweep(args.config)

    # Expand to list of configs (dicts)
    grid = _expand_grid(sweep_space)
    if args.limit is not None:
        grid = grid[: args.limit]

    # Make sure output dir exists
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # CSV header
    fieldnames = [
        "timestamp",
        "device",
        "gpu_name",
        "batch_size",
        "gpu_resident",
        "precision_requested",
        "precision_effective",
        "width",
        "depth",
        "compile_model",
        "num_workers",
        "optimizer",
        "base_lr",
        "weight_decay",
        "warmup_steps",
        "measure_steps",
        "samples_per_sec",
        "elapsed_s",
        "success",
        "error",
        "params_million",
        "gpu_util_avg_pct",
        "gpu_util_max_pct",
        "gpu_mem_used_avg_GiB",
        "gpu_mem_used_max_GiB",
        "alloc_peak_GiB",
        "reserved_peak_GiB",
    ]

    # Run sweep
    rows: List[Dict[str, Any]] = []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, cfg in enumerate(grid, 1):
            result = run_once(cfg, use_gpu=use_gpu)
            writer.writerow(result)
            f.flush()
            rows.append(result)
            print(
                f"[{i:>4}/{len(grid)}] {result['device']} {result['gpu_name']} "
                f"bsz={result['batch_size']} width={result['width']} "
                f"workers={result['num_workers']} prec={result['precision_effective']} "
                f"gpu_res={result['gpu_resident']} -> {result['samples_per_sec']:.1f} samples/s "
                f"{'OK' if result['success'] else 'FAIL'}"
            )

    # Also print top-5 by throughput
    top = sorted(rows, key=lambda r: r["samples_per_sec"], reverse=True)[:5]
    print("\nTop 5 configs by samples/s:")
    for r in top:
        print(
            f" {r['samples_per_sec']:.1f} sps | bsz={r['batch_size']} width={r['width']} "
            f"prec={r['precision_effective']} gpu_res={r['gpu_resident']} "
            f"workers={r['num_workers']} compile={r['compile_model']}"
        )


if __name__ == "__main__":
    main()
