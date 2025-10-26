#!/usr/bin/env python3
"""Analyse MNIST implicit regularisation sweeps by estimating λ for R_IG.

This script loads the locally cached W&B runs belonging to a sweep, restores
the trained MNIST models, and fits the GradientSquaredPenaltyEstimator to
recover the implicit gradient regularisation strength λ.  Results are collated
into a table and a set of diagnostic plots comparing the learned λ values
against the theoretical prediction λ = (h · m) / 4, where h is the learning
rate and m the number of model parameters.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.wandb_utils import (  # noqa: E402
    get_sweep_run_paths,
    load_run_config,
)
from core.estimators import GradientSquaredPenaltyEstimator  # noqa: E402
from experiments.mnist_implicit_reg import (  # noqa: E402
    MNISTImplicitDataModule,
    MNISTImplicitModule,
    RunConfig,
)

STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_FIGURE_DIR = REPO_ROOT / "analysis" / "figures"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "analysis"
MODEL_DIR = REPO_ROOT / "saved_models" / "mnist_implicit_reg"
RESULTS_DIR = REPO_ROOT / "results" / "mnist_implicit_reg"


@dataclass(slots=True)
class RunAnalysis:
    run_id: str
    width: int
    depth: int
    learning_rate: float
    batch_size: int
    num_params: int
    lambda_theoretical: float
    lambda_estimated: float
    relative_error: float
    checkpoint_path: Path
    summary_path: Path | None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional W&B sweep identifier (e.g. yb75ufyh).",
    )
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        default=None,
        help="Limit analysis to specific run id(s). Repeat flag to analyse multiple runs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Analyse a standalone checkpoint (skips W&B sweep lookup).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path to associate with --checkpoint-path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help="Limit the number of training samples used for the estimator (0 = full dataset).",
    )
    parser.add_argument(
        "--estimator-epochs",
        type=int,
        default=50,
        help="Training epochs for the GradientSquaredPenaltyEstimator.",
    )
    parser.add_argument(
        "--estimator-lr",
        type=float,
        default=5e-2,
        help="Learning rate for the GradientSquaredPenaltyEstimator optimiser.",
    )
    parser.add_argument(
        "--estimator-batch-size",
        type=int,
        default=1024,
        help="Batch size used when fitting the estimator.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=None,
        help="Optional CSV path for saving the summary table.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path (defaults to logs/analysis/<timestamp>.log).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Print the results table to stdout in addition to saving it.",
    )
    args = parser.parse_args(argv)
    if args.sweep_id is None and args.checkpoint_path is None:
        parser.error("Either --sweep-id or --checkpoint-path must be provided.")
    if args.summary_path is not None and args.checkpoint_path is None:
        parser.error("--summary-path requires --checkpoint-path.")
    return args


def setup_logging(log_level: str, log_file: Path | None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = DEFAULT_LOG_DIR / f"mnist_rig_lambda_{timestamp}.log"
    log_file = log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    logging.getLogger(__name__).info("Logging to %s", log_file)


def apply_matplotlib_style() -> None:
    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)
    else:
        logging.getLogger(__name__).warning("Matplotlib style %s not found; using default style.", STYLE_PATH)


def unwrap_wandb_value(entry: Any, *, default: Any = None) -> Any:
    if entry is None:
        return default
    if isinstance(entry, dict):
        if "value" in entry:
            return entry["value"]
        return {k: unwrap_wandb_value(v) for k, v in entry.items()}
    return entry


def fetch_config_value(config: dict[str, Any], *candidates: str, default: Any = None) -> Any:
    for key in candidates:
        if key in config:
            value = unwrap_wandb_value(config[key])
            if value in ("null", ""):
                return default
            return value
    return default


def build_run_config(payload: dict[str, Any]) -> RunConfig:
    cfg = RunConfig()
    cfg.width = int(fetch_config_value(payload, "width", default=cfg.width))
    cfg.depth = int(fetch_config_value(payload, "depth", default=cfg.depth))
    cfg.learning_rate = float(
        fetch_config_value(payload, "learning_rate", "learning-rate", default=cfg.learning_rate)
    )
    cfg.batch_size = int(fetch_config_value(payload, "batch_size", "batch-size", default=cfg.batch_size))
    cfg.max_epochs = int(fetch_config_value(payload, "max_epochs", "max-epochs", default=cfg.max_epochs))
    cfg.weight_decay = float(fetch_config_value(payload, "weight_decay", default=cfg.weight_decay))
    cfg.momentum = float(fetch_config_value(payload, "momentum", default=cfg.momentum))
    cfg.nesterov = bool(fetch_config_value(payload, "nesterov", default=cfg.nesterov))
    cfg.seed = int(fetch_config_value(payload, "seed", default=cfg.seed))
    cfg.num_workers = int(fetch_config_value(payload, "num_workers", default=cfg.num_workers))
    cfg.device = str(fetch_config_value(payload, "device", default=cfg.device))
    cfg.log_interval = int(fetch_config_value(payload, "log_interval", default=cfg.log_interval))
    cfg.patience = fetch_config_value(payload, "patience", default=cfg.patience)
    cfg.wandb_project = fetch_config_value(payload, "wandb_project", "wandb-project", default=cfg.wandb_project)
    cfg.wandb_entity = fetch_config_value(payload, "wandb_entity", default=cfg.wandb_entity)
    cfg.wandb_group = fetch_config_value(payload, "wandb_group", default=cfg.wandb_group)
    wandb_tags = fetch_config_value(payload, "wandb_tags", default=cfg.wandb_tags)
    if isinstance(wandb_tags, list):
        cfg.wandb_tags = tuple(wandb_tags)
    else:
        cfg.wandb_tags = wandb_tags
    cfg.wandb_mode = fetch_config_value(payload, "wandb_mode", default=cfg.wandb_mode)
    cfg.wandb_run_name = fetch_config_value(payload, "wandb_run_name", default=cfg.wandb_run_name)
    cfg.wandb_watch = bool(fetch_config_value(payload, "wandb_watch", default=cfg.wandb_watch))
    cfg.wandb_log_model = bool(fetch_config_value(payload, "wandb_log_model", default=cfg.wandb_log_model))
    cfg.wandb_save_code = bool(fetch_config_value(payload, "wandb_save_code", default=cfg.wandb_save_code))
    cfg.output_path = None
    cfg.checkpoint_path = None
    mnist_root = fetch_config_value(payload, "mnist_root", default=str(cfg.mnist_root))
    cfg.mnist_root = Path(mnist_root)
    cfg.mnist_download = bool(fetch_config_value(payload, "mnist_download", default=False))
    cfg.max_train_batches = fetch_config_value(payload, "max_train_batches", default=cfg.max_train_batches)
    cfg.max_eval_batches = fetch_config_value(payload, "max_eval_batches", default=cfg.max_eval_batches)
    cfg.accumulate_rig = bool(fetch_config_value(payload, "accumulate_rig", default=cfg.accumulate_rig))
    return cfg


def format_learning_rate(value: float) -> str:
    return format(value, "g")


def resolve_checkpoint_path(width: int, learning_rate: float, run_id: str) -> Path | None:
    lr_fragment = format_learning_rate(learning_rate)
    direct = MODEL_DIR / f"mnist_width{width}_lr{lr_fragment}_{run_id}.pt"
    if direct.exists():
        return direct
    pattern = f"mnist_width{width}_lr{lr_fragment}_*.pt"
    candidates = sorted(MODEL_DIR.glob(pattern))
    if not candidates:
        logging.getLogger(__name__).warning(
            "No checkpoint found for width=%d lr=%s (run %s)", width, lr_fragment, run_id
        )
        return None
    if len(candidates) == 1:
        return candidates[0]
    logging.getLogger(__name__).warning(
        "Multiple checkpoints match width=%d lr=%s; using the latest for run %s",
        width,
        lr_fragment,
        run_id,
    )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_summary_path(width: int, learning_rate: float, run_id: str) -> Path | None:
    lr_fragment = format_learning_rate(learning_rate)
    candidate = RESULTS_DIR / f"mnist_width{width}_lr{lr_fragment}_{run_id}.json"
    if candidate.exists():
        return candidate
    pattern = f"mnist_width{width}_lr{lr_fragment}_*.json"
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        logger.error("Checkpoint %s does not exist.", checkpoint_path)
        return {}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - torch.load failure
        logger.error("Failed to load checkpoint %s: %s", checkpoint_path, exc)
        return {}
    if not isinstance(payload, dict):
        logger.error("Unexpected checkpoint structure at %s (expected dict).", checkpoint_path)
        return {}
    return payload


def collect_tensor_dataset(
    datamodule: MNISTImplicitDataModule,
    max_samples: int,
    batch_size: int,
) -> TensorDataset:
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataset = datamodule.train_dataset
    if dataset is None:
        raise RuntimeError("MNIST dataset failed to load from the data module.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    collected = 0

    for inputs, labels in loader:
        features.append(inputs)
        targets.append(labels)
        collected += inputs.size(0)
        if max_samples > 0 and collected >= max_samples:
            break

    feature_tensor = torch.cat(features, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    if max_samples > 0:
        feature_tensor = feature_tensor[:max_samples]
        target_tensor = target_tensor[:max_samples]
    return TensorDataset(feature_tensor, target_tensor)


def estimate_lambda(
    predictive_model: nn.Module,
    dataset: TensorDataset,
    learning_rate: float,
    estimator_lr: float,
    batch_size: int,
    max_epochs: int,
    num_params: int,
) -> float:
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    estimator = GradientSquaredPenaltyEstimator(
        predictive_model=predictive_model,
        predictive_loss_fn=loss_fn,
        lr=estimator_lr,
        gd_step_size=learning_rate,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": 1,
        "logger": True,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "log_every_n_steps": 10,
        "deterministic": True,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(estimator, train_dataloaders=dataloader)
    with torch.no_grad():
        lambda_per_param = estimator.bias_model().detach().cpu().item()
    return lambda_per_param * num_params


def compute_run_analysis(
    run_id: str,
    run_cfg: RunConfig,
    checkpoint_path: Path,
    *,
    checkpoint_payload: dict[str, Any] | None,
    summary_path: Path | None,
    args: argparse.Namespace,
) -> RunAnalysis | None:
    logger = logging.getLogger(__name__)
    logger.info("Loading checkpoint for run %s from %s", run_id, checkpoint_path)
    checkpoint_payload = checkpoint_payload or load_checkpoint_payload(checkpoint_path)
    print(checkpoint_payload.keys())
    if not checkpoint_payload:
        return None

    state_dict = checkpoint_payload.get("state_dict")
    print(state_dict)
    if state_dict is None:
        logger.error("Checkpoint %s is missing a 'state_dict' entry.", checkpoint_path)
        return None

    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime GPU availability
        torch.cuda.manual_seed_all(run_cfg.seed)

    module = MNISTImplicitModule(run_cfg)
    module.eval()
    module.requires_grad_(False)
    module.load_state_dict(state_dict)
    predictive_model = module.model
    print(type(predictive_model))
    print(sum(p.numel() for p in predictive_model.parameters()))
    predictive_model.eval()
    predictive_model.requires_grad_(False)

    datamodule = MNISTImplicitDataModule(run_cfg)
    dataset = collect_tensor_dataset(
        datamodule=datamodule,
        max_samples=args.max_samples,
        batch_size=run_cfg.batch_size,
    )
    dataset_size = len(dataset)
    if dataset_size == 0:
        logger.error("Dataset for run %s is empty; skipping analysis.", run_id)
        return None

    lambda_estimated = estimate_lambda(
        predictive_model=predictive_model,
        dataset=dataset,
        learning_rate=run_cfg.learning_rate,
        estimator_lr=args.estimator_lr,
        batch_size=min(args.estimator_batch_size, dataset_size),
        max_epochs=args.estimator_epochs,
    )
    num_params = sum(p.numel() for p in predictive_model.parameters() if p.requires_grad)
    lambda_theoretical = run_cfg.learning_rate * num_params / 4.0
    relative_error = abs(lambda_estimated - lambda_theoretical) / max(abs(lambda_theoretical), 1e-12)

    logger.info(
        "Run %s | width=%d depth=%d lr=%.5g | λ_theory=%.4f | λ_est=%.4f | rel_err=%.3f%%",
        run_id,
        run_cfg.width,
        run_cfg.depth,
        run_cfg.learning_rate,
        lambda_theoretical,
        lambda_estimated,
        relative_error * 100.0,
    )

    return RunAnalysis(
        run_id=run_id,
        width=run_cfg.width,
        depth=run_cfg.depth,
        learning_rate=run_cfg.learning_rate,
        batch_size=run_cfg.batch_size,
        num_params=num_params,
        lambda_theoretical=lambda_theoretical,
        lambda_estimated=lambda_estimated,
        relative_error=relative_error,
        checkpoint_path=checkpoint_path.resolve(),
        summary_path=summary_path.resolve() if summary_path is not None else None,
    )


def analyse_run(
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> RunAnalysis | None:
    logger = logging.getLogger(__name__)
    config_payload = load_run_config(run_dir)
    if not config_payload:
        logger.warning("Skipping run %s (missing or unreadable config).", run_id)
        return None

    run_cfg = build_run_config(config_payload)
    checkpoint_path = resolve_checkpoint_path(run_cfg.width, run_cfg.learning_rate, run_id)
    if checkpoint_path is None:
        logger.warning("Skipping run %s (checkpoint unavailable).", run_id)
        return None

    summary_path = resolve_summary_path(run_cfg.width, run_cfg.learning_rate, run_id)

    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    if not checkpoint_payload:
        logger.warning("Skipping run %s (checkpoint payload unreadable).", run_id)
        return None

    return compute_run_analysis(
        run_id=run_id,
        run_cfg=run_cfg,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        summary_path=summary_path,
        args=args,
    )


def write_results_table(results: list[RunAnalysis], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_id",
        "width",
        "depth",
        "learning_rate",
        "batch_size",
        "num_params",
        "lambda_theoretical",
        "lambda_estimated",
        "relative_error",
        "checkpoint_path",
        "summary_path",
    ]
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for item in results:
            writer.writerow(
                [
                    item.run_id,
                    item.width,
                    item.depth,
                    f"{item.learning_rate:.8g}",
                    item.batch_size,
                    item.num_params,
                    f"{item.lambda_theoretical:.6f}",
                    f"{item.lambda_estimated:.6f}",
                    f"{item.relative_error:.6e}",
                    item.checkpoint_path,
                    item.summary_path or "",
                ]
            )
    logging.getLogger(__name__).info("Saved results table to %s", destination)


def print_results_table(results: list[RunAnalysis]) -> None:
    headers = [
        ("run_id", max(len("run_id"), max(len(r.run_id) for r in results))),
        ("width", len("width")),
        ("depth", len("depth")),
        ("learning_rate", len("learning_rate")),
        ("num_params", len("num_params")),
        ("lambda_theoretical", len("lambda_theoretical")),
        ("lambda_estimated", len("lambda_estimated")),
        ("relative_error_pct", len("relative_error_pct")),
    ]
    header_line = " | ".join(label.ljust(width) for label, width in headers)
    separator = "-+-".join("-" * width for _, width in headers)
    print(header_line)
    print(separator)
    for result in results:
        row = [
            result.run_id.ljust(headers[0][1]),
            f"{result.width:d}".ljust(headers[1][1]),
            f"{result.depth:d}".ljust(headers[2][1]),
            f"{result.learning_rate:.5g}".ljust(headers[3][1]),
            f"{result.num_params:d}".ljust(headers[4][1]),
            f"{result.lambda_theoretical:.4f}".ljust(headers[5][1]),
            f"{result.lambda_estimated:.4f}".ljust(headers[6][1]),
            f"{result.relative_error * 100.0:>8.3f}".rjust(headers[7][1]),
        ]
        print(" | ".join(row))


def generate_plots(results: list[RunAnalysis], figure_dir: Path, sweep_id: str) -> None:
    figure_dir = figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    theoretical = [item.lambda_theoretical for item in results]
    estimated = [item.lambda_estimated for item in results]
    num_params = [item.num_params for item in results]

    # Scatter: theoretical vs estimated λ
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
    scatter = ax_scatter.scatter(
        theoretical,
        estimated,
        c=[math.log10(max(n, 1)) for n in num_params],
        cmap="viridis",
        edgecolor="none",
    )
    max_lambda = max(max(theoretical), max(estimated))
    ax_scatter.plot([0, max_lambda], [0, max_lambda], linestyle="--", color="black", linewidth=1.0)
    ax_scatter.set_xlabel("λ (theoretical)")
    ax_scatter.set_ylabel("λ (estimated)")
    ax_scatter.set_title(f"R_IG λ estimates for sweep {sweep_id}")
    cbar = fig_scatter.colorbar(scatter, ax=ax_scatter, label="log10(num_params)")
    cbar.ax.set_ylabel("log₁₀(# parameters)", rotation=270, labelpad=18)
    scatter_path = figure_dir / f"{sweep_id}_lambda_scatter.png"
    fig_scatter.tight_layout()
    fig_scatter.savefig(scatter_path, dpi=200)
    plt.close(fig_scatter)
    logging.getLogger(__name__).info("Saved scatter plot to %s", scatter_path)

    # Line plot: λ vs learning rate grouped by width.
    grouped: defaultdict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for item in results:
        grouped[item.width].append(
            (item.learning_rate, item.lambda_theoretical, item.lambda_estimated)
        )

    fig_line, ax_line = plt.subplots(figsize=(6, 5))
    for width, entries in sorted(grouped.items()):
        entries.sort(key=lambda triple: triple[0])
        rates = [e[0] for e in entries]
        theo_vals = [e[1] for e in entries]
        est_vals = [e[2] for e in entries]
        ax_line.plot(rates, theo_vals, marker="o", linestyle="--", label=f"width {width} (theory)")
        ax_line.plot(rates, est_vals, marker="s", linestyle="-", label=f"width {width} (est.)")
    ax_line.set_xlabel("Learning rate")
    ax_line.set_ylabel("λ")
    ax_line.set_title("λ vs. learning rate by width")
    ax_line.set_xscale("log")
    ax_line.legend(ncol=2, fontsize="small")
    fig_line.tight_layout()
    line_path = figure_dir / f"{sweep_id}_lambda_vs_lr.png"
    fig_line.savefig(line_path, dpi=200)
    plt.close(fig_line)
    logging.getLogger(__name__).info("Saved λ vs learning-rate plot to %s", line_path)

    # Relative error vs number of parameters.
    fig_error, ax_error = plt.subplots(figsize=(6, 4))
    ax_error.scatter(num_params, [item.relative_error * 100.0 for item in results], color="#d62728")
    ax_error.set_xscale("log")
    ax_error.set_xlabel("Number of parameters")
    ax_error.set_ylabel("Relative error (%)")
    ax_error.set_title("Relative error in λ estimate")
    fig_error.tight_layout()
    error_path = figure_dir / f"{sweep_id}_lambda_relative_error.png"
    fig_error.savefig(error_path, dpi=200)
    plt.close(fig_error)
    logging.getLogger(__name__).info("Saved relative error plot to %s", error_path)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    apply_matplotlib_style()

    results: list[RunAnalysis] = []

    run_id_filter = list(dict.fromkeys(args.run_ids)) if args.run_ids else []

    if args.checkpoint_path is not None:
        if run_id_filter and len(run_id_filter) > 1:
            logging.getLogger(__name__).warning(
                "Multiple --run-id values provided with --checkpoint-path; using the first (%s).",
                run_id_filter[0],
            )
        run_id = run_id_filter[0] if run_id_filter else args.checkpoint_path.stem
        checkpoint_payload = load_checkpoint_payload(args.checkpoint_path)
        if not checkpoint_payload:
            logging.getLogger(__name__).error("Aborting due to checkpoint load failure.")
            sys.exit(1)
        config_payload = checkpoint_payload.get("config")
        if not config_payload:
            logging.getLogger(__name__).error(
                "Checkpoint %s is missing the stored training config.",
                args.checkpoint_path,
            )
            sys.exit(1)
        run_cfg = build_run_config(config_payload)
        summary_path = args.summary_path
        if summary_path is None:
            summary_path = resolve_summary_path(run_cfg.width, run_cfg.learning_rate, run_id)
        try:
            analysis = compute_run_analysis(
                run_id=run_id,
                run_cfg=run_cfg,
                checkpoint_path=args.checkpoint_path,
                checkpoint_payload=checkpoint_payload,
                summary_path=summary_path,
                args=args,
            )
        except Exception:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).exception("Failed to analyse checkpoint %s", args.checkpoint_path)
            analysis = None
        if analysis is not None:
            results.append(analysis)
    else:
        run_paths = get_sweep_run_paths(args.sweep_id)
        if not run_paths:
            logging.getLogger(__name__).error(
                "No W&B runs discovered for sweep %s. Check the sweep ID or local W&B cache.",
                args.sweep_id,
            )
            sys.exit(1)

        if run_id_filter:
            missing_ids = [run_id for run_id in run_id_filter if run_id not in run_paths]
            for run_id in missing_ids:
                logging.getLogger(__name__).warning(
                    "Requested run id %s not found in sweep %s.", run_id, args.sweep_id
                )
            run_paths = {run_id: run_paths[run_id] for run_id in run_id_filter if run_id in run_paths}
            if not run_paths:
                logging.getLogger(__name__).error("No remaining runs to analyse after filtering; aborting.")
                sys.exit(1)

        for run_id, run_dir in sorted(run_paths.items()):
            try:
                analysis = analyse_run(run_id, run_dir, args)
            except Exception:  # pragma: no cover - defensive logging
                logging.getLogger(__name__).exception("Failed to analyse run %s", run_id)
                continue
            if analysis is not None:
                results.append(analysis)

    if not results:
        logging.getLogger(__name__).error("No runs produced results; aborting.")
        sys.exit(2)

    results.sort(key=lambda item: (item.width, item.learning_rate, item.run_id))

    if args.output_table is not None:
        write_results_table(results, args.output_table)
    if args.pretty_print:
        print_results_table(results)
    label_for_plots = args.sweep_id if args.sweep_id is not None else results[0].run_id
    generate_plots(results, args.figure_dir, label_for_plots)


if __name__ == "__main__":
    main()
