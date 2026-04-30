#!/usr/bin/env python3
"""Estimate scalar ridge strength for an OLMo checkpoint on a wiki data subset."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_REVISION = "stage1-step1907359-tokens4001B"
DEFAULT_DATASET_REPO = "allenai/olmo-mix-1124"
DEFAULT_DATASET_FILE = "data/wiki/wiki-0001.json.gz"
DEFAULT_HF_HOME = Path("/shared_data0/jrudoler/.cache/huggingface")
DEFAULT_GRADIENT_ROOT = Path("/shared_data0/jrudoler/inductive-bias/olmo_ridge_estimation")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "generated"
    / "olmo_ridge_estimation"
    / "olmo2_1b_stage1_wiki0001_50m.json"
)

NORM_NAME_PARTS = ("norm", "layernorm", "layer_norm", ".ln", "_ln")
DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


@dataclass(frozen=True)
class ScopeStats:
    """Sufficient statistics for closed-form scalar ridge estimation."""

    param_count: int = 0
    theta_dot_grad: float = 0.0
    theta_sq_norm: float = 0.0
    grad_sq_norm: float = 0.0

    @property
    def lambda_hat(self) -> float:
        if self.theta_sq_norm <= 0.0:
            return float("nan")
        return -self.theta_dot_grad / (2.0 * self.theta_sq_norm)

    @property
    def lambda_hat_nonnegative(self) -> float:
        value = self.lambda_hat
        return max(value, 0.0) if math.isfinite(value) else value

    @property
    def cosine_alignment(self) -> float:
        denom = math.sqrt(self.theta_sq_norm * self.grad_sq_norm)
        if denom <= 0.0:
            return float("nan")
        return -self.theta_dot_grad / denom

    @property
    def residual_ratio(self) -> float:
        lam = self.lambda_hat
        if self.grad_sq_norm <= 0.0 or not math.isfinite(lam):
            return float("nan")
        residual_sq = (
            self.grad_sq_norm
            + 4.0 * lam * self.theta_dot_grad
            + 4.0 * lam * lam * self.theta_sq_norm
        )
        return math.sqrt(max(residual_sq, 0.0) / self.grad_sq_norm)

    def to_json_dict(self, predicted_tokens: int | None = None) -> dict[str, float | int]:
        lambda_sum_loss = (
            self.lambda_hat * predicted_tokens if predicted_tokens is not None else float("nan")
        )
        return {
            **asdict(self),
            "lambda_hat": self.lambda_hat,
            "lambda_hat_nonnegative": self.lambda_hat_nonnegative,
            "lambda_mean_loss": self.lambda_hat,
            "lambda_sum_loss": lambda_sum_loss,
            "weight_decay_equivalent_sum_loss": 2.0 * lambda_sum_loss,
            "cosine_alignment": self.cosine_alignment,
            "residual_ratio": self.residual_ratio,
        }


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    parameter: torch.nn.Parameter
    group: str
    decay_eligible: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=50_000_000,
        help="Maximum raw token count to pack from the dataset file.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Packed causal-LM input length. Incomplete final blocks are dropped.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device for the model and gradient computation.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=("auto", "float32", "bfloat16", "float16"),
        help=(
            "Model compute dtype. Defaults to float32 for gradient-accumulation "
            "accuracy; 'auto' uses bfloat16 on CUDA and float32 on CPU."
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)),
        help="Hugging Face cache root for model and dataset downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path for scalar ridge estimates and diagnostics.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional torch .pt output path. Defaults to output with .pt suffix.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument(
        "--save-gradients",
        action="store_true",
        help="Save reusable per-parameter loss gradients after estimation.",
    )
    parser.add_argument(
        "--gradient-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for saved gradient shards and manifest. Defaults under "
            f"{DEFAULT_GRADIENT_ROOT} to avoid the /home quota."
        ),
    )
    parser.add_argument(
        "--gradient-save-dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Dtype used when writing gradient tensors to disk.",
    )
    parser.add_argument(
        "--gradient-shard-max-params",
        type=int,
        default=64_000_000,
        help="Approximate maximum number of scalar gradient entries per shard.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to Hugging Face model/tokenizer loaders.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(value)


def resolve_torch_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[value]


def dtype_name(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype, str(dtype).removeprefix("torch."))


def iter_jsonl_texts_from_gzip(path: Path) -> Iterator[str]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            text = payload.get("text")
            if isinstance(text, str) and text:
                yield text


def _take_token_budget(
    tokens: Sequence[int],
    *,
    remaining_budget: int,
    eos_token_id: int | None,
) -> list[int]:
    if remaining_budget <= 0:
        return []
    sequence = list(tokens)
    if eos_token_id is not None:
        sequence.append(int(eos_token_id))
    return sequence[:remaining_budget]


def pack_token_blocks(
    token_sequences: Iterable[Sequence[int]],
    *,
    sequence_length: int,
    token_budget: int,
    eos_token_id: int | None,
) -> tuple[Tensor, dict[str, int]]:
    """Pack tokenized documents into fixed-length causal-LM blocks."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if token_budget < sequence_length:
        raise ValueError("token_budget must be at least sequence_length.")

    buffer: list[int] = []
    blocks: list[Tensor] = []
    consumed_tokens = 0
    consumed_docs = 0

    for tokens in token_sequences:
        if consumed_tokens >= token_budget:
            break
        chunk = _take_token_budget(
            tokens,
            remaining_budget=token_budget - consumed_tokens,
            eos_token_id=eos_token_id,
        )
        if not chunk:
            continue
        consumed_docs += 1
        consumed_tokens += len(chunk)
        buffer.extend(chunk)
        while len(buffer) >= sequence_length:
            block = buffer[:sequence_length]
            del buffer[:sequence_length]
            blocks.append(torch.tensor(block, dtype=torch.long))

    if not blocks:
        raise RuntimeError(
            "No full token blocks were produced. Increase token_budget or check input data."
        )

    input_ids = torch.stack(blocks)
    metadata = {
        "documents_consumed": consumed_docs,
        "raw_tokens_consumed": consumed_tokens,
        "input_tokens_used": int(input_ids.numel()),
        "blocks": int(input_ids.shape[0]),
        "dropped_tail_tokens": len(buffer),
        "predicted_tokens": int(input_ids.shape[0] * (sequence_length - 1)),
    }
    return input_ids, metadata


def tokenize_texts(
    texts: Iterable[str],
    tokenizer: Any,
) -> Iterator[list[int]]:
    for text in texts:
        tokenized = tokenizer(text, add_special_tokens=False)
        input_ids = tokenized.get("input_ids", [])
        if input_ids:
            yield [int(token_id) for token_id in input_ids]


def download_dataset_file(
    *,
    dataset_repo: str,
    dataset_file: str,
    hf_home: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=dataset_file,
            cache_dir=hf_home,
        )
    )


def load_tokenizer_and_model(
    *,
    model_id: str,
    revision: str,
    hf_home: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool,
    trust_remote_code: bool,
) -> tuple[Any, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=hf_home,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=hf_home,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    return tokenizer, model


def parameter_group(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered or "wte" in lowered or "lm_head" in lowered:
        return "embedding_or_lm_head"
    if any(part in lowered for part in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(part in lowered for part in ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if any(part in lowered for part in NORM_NAME_PARTS):
        return "norm"
    return "other"


def is_decay_eligible_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    if parameter.ndim < 2:
        return False
    if lowered.endswith(".bias") or lowered == "bias":
        return False
    return not any(part in lowered for part in NORM_NAME_PARTS)


def unique_named_parameters(model: torch.nn.Module) -> list[ParameterRecord]:
    records: list[ParameterRecord] = []
    seen: set[tuple[int, int, int]] = set()
    for name, parameter in model.named_parameters():
        key = (parameter.untyped_storage().data_ptr(), parameter.storage_offset(), parameter.numel())
        if key in seen:
            continue
        seen.add(key)
        records.append(
            ParameterRecord(
                name=name,
                parameter=parameter,
                group=parameter_group(name),
                decay_eligible=is_decay_eligible_parameter(name, parameter),
            )
        )
    return records


def _scope_stats_from_records(records: Iterable[ParameterRecord]) -> ScopeStats:
    param_count = 0
    theta_dot_grad = 0.0
    theta_sq_norm = 0.0
    grad_sq_norm = 0.0
    for record in records:
        parameter = record.parameter
        if parameter.grad is None:
            continue
        theta = parameter.detach()
        grad = parameter.grad.detach()
        param_count += parameter.numel()
        theta_dot_grad += float((theta * grad).sum(dtype=torch.float64).cpu())
        theta_sq_norm += float(theta.pow(2).sum(dtype=torch.float64).cpu())
        grad_sq_norm += float(grad.pow(2).sum(dtype=torch.float64).cpu())
    return ScopeStats(
        param_count=param_count,
        theta_dot_grad=theta_dot_grad,
        theta_sq_norm=theta_sq_norm,
        grad_sq_norm=grad_sq_norm,
    )


def compute_parameter_statistics(
    model: torch.nn.Module,
    *,
    predicted_tokens: int | None = None,
) -> dict[str, Any]:
    records = unique_named_parameters(model)
    all_stats = _scope_stats_from_records(records)
    decay_stats = _scope_stats_from_records(record for record in records if record.decay_eligible)

    grouped: dict[str, list[ParameterRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)

    return {
        "all": all_stats.to_json_dict(predicted_tokens),
        "decay_eligible": decay_stats.to_json_dict(predicted_tokens),
        "groups": {
            group: _scope_stats_from_records(group_records).to_json_dict(predicted_tokens)
            for group, group_records in sorted(grouped.items())
        },
        "parameter_names": {
            "decay_eligible": [record.name for record in records if record.decay_eligible],
            "excluded_from_decay": [
                record.name for record in records if not record.decay_eligible
            ],
        },
    }


def _write_gradient_shard(
    *,
    output_dir: Path,
    shard_index: int,
    gradients: dict[str, Tensor],
) -> dict[str, Any]:
    shard_name = f"gradients-{shard_index:05d}.pt"
    shard_path = output_dir / shard_name
    torch.save({"gradients": gradients}, shard_path)
    return {
        "shard_index": shard_index,
        "path": str(shard_path),
        "file": shard_name,
        "num_tensors": len(gradients),
        "numel": int(sum(tensor.numel() for tensor in gradients.values())),
    }


def save_gradient_shards(
    *,
    model: torch.nn.Module,
    output_dir: Path,
    save_dtype: torch.dtype,
    shard_max_params: int,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Save per-parameter loss gradients as sharded torch artifacts."""
    if shard_max_params <= 0:
        raise ValueError("gradient_shard_max_params must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = unique_named_parameters(model)
    manifest: dict[str, Any] = {
        "format": "torch_sharded_parameter_gradients_v1",
        "save_dtype": dtype_name(save_dtype),
        "context": dict(context),
        "shards": [],
        "parameters": {},
    }

    shard_index = 0
    shard_numel = 0
    shard_gradients: dict[str, Tensor] = {}

    for record in records:
        parameter = record.parameter
        if parameter.grad is None:
            continue
        if shard_gradients and shard_numel + parameter.numel() > shard_max_params:
            manifest["shards"].append(
                _write_gradient_shard(
                    output_dir=output_dir,
                    shard_index=shard_index,
                    gradients=shard_gradients,
                )
            )
            shard_index += 1
            shard_numel = 0
            shard_gradients = {}

        grad_cpu = (
            parameter.grad.detach()
            .to(device="cpu", dtype=save_dtype)
            .contiguous()
        )
        shard_gradients[record.name] = grad_cpu
        manifest["parameters"][record.name] = {
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "parameter_dtype": dtype_name(parameter.dtype),
            "gradient_dtype": dtype_name(save_dtype),
            "group": record.group,
            "decay_eligible": record.decay_eligible,
            "shard_index": shard_index,
        }
        shard_numel += parameter.numel()

    if shard_gradients:
        manifest["shards"].append(
            _write_gradient_shard(
                output_dir=output_dir,
                shard_index=shard_index,
                gradients=shard_gradients,
            )
        )

    manifest["num_parameters"] = len(manifest["parameters"])
    manifest["total_numel"] = int(
        sum(item["numel"] for item in manifest["parameters"].values())
    )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Saved gradient manifest %s", manifest_path)
    return {
        "manifest": str(manifest_path),
        "directory": str(output_dir),
        "save_dtype": dtype_name(save_dtype),
        "num_shards": len(manifest["shards"]),
        "num_parameters": manifest["num_parameters"],
        "total_numel": manifest["total_numel"],
        "shards": manifest["shards"],
    }


def accumulate_cross_entropy_gradient(
    *,
    model: torch.nn.Module,
    input_ids: Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, float | int]:
    predicted_tokens = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    batches = 0
    for (batch_input_ids,) in dataloader:
        batch_input_ids = batch_input_ids.to(device=device, non_blocking=True)
        outputs = model(input_ids=batch_input_ids, use_cache=False)
        logits = outputs.logits
        labels = batch_input_ids
        loss_sum = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_loss_sum += float(loss_sum.detach().cpu())
        (loss_sum / predicted_tokens).backward()
        batches += 1
        if batches % 10 == 0:
            LOGGER.info("Processed %d gradient batches", batches)

    mean_loss = total_loss_sum / predicted_tokens
    return {
        "batches": batches,
        "predicted_tokens": predicted_tokens,
        "cross_entropy": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 100.0 else float("inf"),
    }


def build_payload(
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    token_metadata: Mapping[str, int],
    loss_metadata: Mapping[str, float | int],
    parameter_statistics: Mapping[str, Any],
    gradient_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "config": {
            "model_id": args.model_id,
            "revision": args.revision,
            "dataset_repo": args.dataset_repo,
            "dataset_file": args.dataset_file,
            "dataset_path": str(dataset_path),
            "hf_home": str(args.hf_home),
            "token_budget": args.token_budget,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "save_gradients": args.save_gradients,
        },
        "tokens": dict(token_metadata),
        "loss": dict(loss_metadata),
        "ridge_estimates": dict(parameter_statistics),
    }
    if gradient_artifact is not None:
        payload["gradient_artifact"] = dict(gradient_artifact)
    return payload


def save_outputs(payload: Mapping[str, Any], output: Path, stats_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    torch.save(dict(payload), stats_output)
    LOGGER.info("Saved %s", output)
    LOGGER.info("Saved %s", stats_output)


def main() -> None:
    configure_logging()
    args = parse_args()
    args.hf_home = args.hf_home.expanduser().resolve()
    os.environ["HF_HOME"] = str(args.hf_home)
    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype, device)
    gradient_save_dtype = resolve_torch_dtype(args.gradient_save_dtype, torch.device("cpu"))
    torch.set_float32_matmul_precision("high")
    stats_output = args.stats_output or args.output.with_suffix(".pt")
    gradient_output_dir = args.gradient_output_dir or (
        DEFAULT_GRADIENT_ROOT / "gradients" / args.output.stem
    )
    LOGGER.info("Using device=%s dtype=%s HF_HOME=%s", device, torch_dtype, args.hf_home)

    tokenizer, model = load_tokenizer_and_model(
        model_id=args.model_id,
        revision=args.revision,
        hf_home=args.hf_home,
        device=device,
        torch_dtype=torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
    )
    dataset_path = download_dataset_file(
        dataset_repo=args.dataset_repo,
        dataset_file=args.dataset_file,
        hf_home=args.hf_home,
    )
    LOGGER.info("Using dataset file %s", dataset_path)

    input_ids, token_metadata = pack_token_blocks(
        tokenize_texts(iter_jsonl_texts_from_gzip(dataset_path), tokenizer),
        sequence_length=args.sequence_length,
        token_budget=args.token_budget,
        eos_token_id=tokenizer.eos_token_id,
    )
    LOGGER.info(
        "Packed %d blocks from %d documents (%d input tokens)",
        token_metadata["blocks"],
        token_metadata["documents_consumed"],
        token_metadata["input_tokens_used"],
    )

    loss_metadata = accumulate_cross_entropy_gradient(
        model=model,
        input_ids=input_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    predicted_tokens = int(loss_metadata["predicted_tokens"])
    parameter_statistics = compute_parameter_statistics(
        model,
        predicted_tokens=predicted_tokens,
    )
    gradient_artifact = None
    if args.save_gradients:
        gradient_artifact = save_gradient_shards(
            model=model,
            output_dir=gradient_output_dir,
            save_dtype=gradient_save_dtype,
            shard_max_params=args.gradient_shard_max_params,
            context={
                "model_id": args.model_id,
                "revision": args.revision,
                "dataset_repo": args.dataset_repo,
                "dataset_file": args.dataset_file,
                "token_budget": args.token_budget,
                "sequence_length": args.sequence_length,
                "predicted_tokens": predicted_tokens,
                "loss_normalization": "mean_next_token_cross_entropy",
            },
        )
    payload = build_payload(
        args=args,
        dataset_path=dataset_path,
        token_metadata=token_metadata,
        loss_metadata=loss_metadata,
        parameter_statistics=parameter_statistics,
        gradient_artifact=gradient_artifact,
    )
    save_outputs(payload, args.output, stats_output)


if __name__ == "__main__":
    main()
