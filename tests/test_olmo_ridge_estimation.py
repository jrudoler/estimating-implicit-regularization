from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from analysis.olmo_ridge_estimation.run import (
    ScopeStats,
    compute_parameter_statistics,
    iter_jsonl_texts_from_gzip,
    pack_token_blocks,
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
    assert stats.cosine_alignment == pytest.approx(1.0)
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


def test_iter_jsonl_texts_from_gzip(tmp_path: Path) -> None:
    path = tmp_path / "wiki-mini.json.gz"
    rows = [{"text": "alpha"}, {"metadata": "skip"}, {"text": ""}, {"text": "beta"}]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert list(iter_jsonl_texts_from_gzip(path)) == ["alpha", "beta"]
