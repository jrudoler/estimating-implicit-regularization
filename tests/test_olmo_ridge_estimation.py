from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from analysis.olmo_ridge_estimation.run import (
    ScopeStats,
    accumulate_attention_mlp_chunk_gradients,
    bootstrap_lambdas_from_batch_statistics,
    compute_parameter_statistics,
    iter_jsonl_texts_from_gzip,
    pack_token_blocks,
    save_gradient_shards,
    set_parameter_trainability,
    summarize_chunk_lambdas_from_statistics,
    unique_named_parameters,
)


def test_scope_stats_recovers_closed_form_lambda() -> None:
    theta = torch.tensor([1.0, -2.0, 3.0])
    lam = 0.125
    grad = -2.0 * lam * theta
    stats = ScopeStats(
        param_count=theta.numel(),
        theta_dot_grad=float(torch.dot(theta, grad)),
        theta_sq_norm=float(torch.dot(theta, theta)),
        grad_sq_norm=float(torch.dot(grad, grad)),
    )

    assert stats.lambda_hat == pytest.approx(lam)
    assert stats.lambda_hat_nonnegative == pytest.approx(lam)
    assert stats.to_json_dict(predicted_tokens=10)["lambda_sum_loss"] == pytest.approx(
        10 * lam
    )
    assert stats.to_json_dict(predicted_tokens=10)[
        "weight_decay_equivalent_sum_loss"
    ] == pytest.approx(20 * lam)
    assert stats.cosine_alignment == pytest.approx(1.0)
    assert stats.projection_r2 == pytest.approx(1.0)
    assert stats.residual_ratio == pytest.approx(0.0)


class TiedToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 3)
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed.weight
        self.norm = nn.LayerNorm(3)
        self.bias = nn.Parameter(torch.ones(3))


def test_parameter_filtering_deduplicates_tied_weights() -> None:
    model = TiedToyModel()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    stats = compute_parameter_statistics(model)

    assert stats["all"]["param_count"] == 21
    assert stats["decay_eligible"]["param_count"] == 12
    assert stats["parameter_names"]["decay_eligible"] == ["embed.weight"]
    assert set(stats["parameter_names"]["excluded_from_decay"]) == {
        "bias",
        "norm.weight",
        "norm.bias",
    }


class BlockToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 1, bias=False)
        self.embed = nn.Embedding(2, 2)
        self.norm = nn.LayerNorm(2)


def test_attention_plus_mlp_scope_sums_sufficient_statistics() -> None:
    model = BlockToyModel()
    with torch.no_grad():
        model.self_attn.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.mlp.weight.copy_(torch.tensor([[5.0, 6.0]]))
    model.self_attn.weight.grad = -0.2 * model.self_attn.weight.detach()
    model.mlp.weight.grad = -0.6 * model.mlp.weight.detach()
    model.embed.weight.grad = torch.ones_like(model.embed.weight)
    model.norm.weight.grad = torch.ones_like(model.norm.weight)
    model.norm.bias.grad = torch.ones_like(model.norm.bias)

    stats = compute_parameter_statistics(model)
    attention = stats["groups"]["attention"]
    mlp = stats["groups"]["mlp"]
    combined = stats["attention_plus_mlp"]
    expected = -(
        attention["theta_dot_grad"] + mlp["theta_dot_grad"]
    ) / (2.0 * (attention["theta_sq_norm"] + mlp["theta_sq_norm"]))

    assert combined["param_count"] == 6
    assert combined["lambda_hat"] == pytest.approx(expected)
    assert combined["lambda_hat"] != pytest.approx(
        (attention["lambda_hat"] + mlp["lambda_hat"]) / 2.0
    )


def test_pack_token_blocks_is_deterministic() -> None:
    blocks, metadata = pack_token_blocks(
        [[1, 2, 3], [4], [5, 6, 7, 8]],
        sequence_length=4,
        token_budget=10,
        eos_token_id=0,
    )

    assert blocks.tolist() == [[1, 2, 3, 0], [4, 0, 5, 6]]
    assert metadata == {
        "documents_consumed": 3,
        "raw_tokens_consumed": 10,
        "input_tokens_used": 8,
        "blocks": 2,
        "dropped_tail_tokens": 2,
        "predicted_tokens": 6,
    }


def test_pack_token_blocks_accepts_full_source_budget() -> None:
    blocks, metadata = pack_token_blocks(
        [[1, 2], [3, 4, 5]],
        sequence_length=3,
        token_budget=None,
        eos_token_id=0,
    )

    assert blocks.tolist() == [[1, 2, 0], [3, 4, 5]]
    assert metadata["raw_tokens_consumed"] == 7
    assert metadata["input_tokens_used"] == 6


def test_batch_bootstrap_uses_summed_sufficient_statistics() -> None:
    result = bootstrap_lambdas_from_batch_statistics(
        batch_theta_dot_grad_sum_loss={
            "attention": [-4.0, -8.0],
            "mlp": [-2.0, -4.0],
            "attention_plus_mlp": [-6.0, -12.0],
        },
        batch_predicted_tokens=[10, 10],
        theta_sq_norm_by_scope={
            "attention": 2.0,
            "mlp": 1.0,
            "attention_plus_mlp": 3.0,
        },
        n_replicates=5,
        seed=0,
    )

    assert set(result) == {"attention", "mlp", "attention_plus_mlp"}
    assert result["attention"]["n"] == 5
    assert result["attention_plus_mlp"]["std"] >= 0.0
    assert min(result["attention_plus_mlp"]["samples"]) >= 0.1
    assert max(result["attention_plus_mlp"]["samples"]) <= 0.2


def test_chunk_estimates_report_mean_and_sem() -> None:
    result = summarize_chunk_lambdas_from_statistics(
        chunk_theta_dot_grad_sum_loss={
            "attention": [-4.0, -8.0, -12.0],
            "mlp": [-2.0, -4.0, -6.0],
            "attention_plus_mlp": [-6.0, -12.0, -18.0],
        },
        chunk_predicted_tokens=[10, 10, 10],
        chunk_batches=[5, 5, 5],
        theta_sq_norm_by_scope={
            "attention": 2.0,
            "mlp": 1.0,
            "attention_plus_mlp": 3.0,
        },
        target_predicted_tokens=10,
        dropped_tail_predicted_tokens=3,
        dropped_tail_batches=1,
    )

    combined = result["attention_plus_mlp"]
    assert combined["unit"] == "contiguous_token_chunk"
    assert combined["n"] == 3
    assert combined["mean"] == pytest.approx(0.2)
    assert combined["sem"] > 0.0
    assert combined["chunk_predicted_tokens"] == [10, 10, 10]
    assert combined["dropped_tail_predicted_tokens"] == 3


class TinyCausalAttentionMlpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.self_attn = nn.Linear(3, 3, bias=False)
        self.mlp = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 5, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del use_cache
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.self_attn(hidden) + self.mlp(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_chunk_gradient_accumulation_retains_exact_fit_statistics() -> None:
    torch.manual_seed(7)
    model = TinyCausalAttentionMlpModel()
    records = unique_named_parameters(model)
    set_parameter_trainability(records, parameter_scope="attention_mlp")
    input_ids = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 0],
            [4, 0, 1],
        ]
    )

    loss_metadata, estimates = accumulate_attention_mlp_chunk_gradients(
        model=model,
        records=records,
        input_ids=input_ids,
        batch_size=1,
        num_workers=0,
        device=torch.device("cpu"),
        chunk_token_budget=4,
    )

    combined = estimates["attention_plus_mlp"]["chunk_token_estimates"]
    first_chunk = combined["gradient_statistics"][0]
    assert combined["n"] == 2
    assert combined["chunk_predicted_tokens"] == [4, 4]
    assert loss_metadata["token_chunks"]["dropped_tail_predicted_tokens"] == 2
    assert len(loss_metadata["token_chunks"]["chunk_cross_entropy"]) == 2
    assert first_chunk["cosine_alignment"] == pytest.approx(
        combined["cosine_alignment_samples"][0]
    )
    assert first_chunk["projection_r2"] == pytest.approx(
        first_chunk["cosine_alignment"] ** 2
    )
    assert first_chunk["residual_ratio"] == pytest.approx(
        (1.0 - first_chunk["projection_r2"]) ** 0.5
    )

    model.zero_grad(set_to_none=True)
    direct_outputs = model(input_ids=input_ids[:2], use_cache=False)
    direct_loss = nn.functional.cross_entropy(
        direct_outputs.logits[:, :-1, :].contiguous().view(-1, 5),
        input_ids[:2, 1:].contiguous().view(-1),
        reduction="sum",
    )
    (direct_loss / 4).backward()
    selected_records = [
        record for record in records if record.group in {"attention", "mlp"}
    ]
    expected_dot = sum(
        float(
            (record.parameter.detach() * record.parameter.grad.detach())
            .sum(dtype=torch.float64)
            .cpu()
        )
        for record in selected_records
    )
    expected_grad_sq = sum(
        float(
            record.parameter.grad.detach()
            .pow(2)
            .sum(dtype=torch.float64)
            .cpu()
        )
        for record in selected_records
    )
    assert first_chunk["theta_dot_grad"] == pytest.approx(expected_dot)
    assert first_chunk["grad_sq_norm"] == pytest.approx(expected_grad_sq)


def test_iter_jsonl_texts_from_gzip(tmp_path: Path) -> None:
    path = tmp_path / "wiki-mini.json.gz"
    rows = [{"text": "alpha"}, {"metadata": "skip"}, {"text": ""}, {"text": "beta"}]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert list(iter_jsonl_texts_from_gzip(path)) == ["alpha", "beta"]


def test_save_gradient_shards_writes_manifest(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    artifact = save_gradient_shards(
        model=model,
        output_dir=tmp_path / "grads",
        save_dtype=torch.float32,
        shard_max_params=4,
        context={"model_id": "toy"},
    )

    manifest = json.loads(Path(artifact["manifest"]).read_text())
    assert manifest["format"] == "torch_sharded_parameter_gradients_v1"
    assert manifest["num_parameters"] == 2
    assert manifest["total_numel"] == 8
    assert len(manifest["shards"]) == 2

    first_shard = torch.load(manifest["shards"][0]["path"], weights_only=False)
    assert "weight" in first_shard["gradients"]
