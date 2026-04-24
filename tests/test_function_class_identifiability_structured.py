from __future__ import annotations

import math

import torch

from analysis.function_class_identifiability.run import (
    build_spectrum_values,
    build_teacher_first_layer,
    generate_dataset,
)


def test_build_spectrum_values_uniform_respects_rank() -> None:
    values = build_spectrum_values(
        size=6,
        active_rank=3,
        family="uniform",
        decay=1.5,
        target_sum=6.0,
    )

    assert torch.allclose(values[:3], torch.full((3,), 2.0))
    assert torch.count_nonzero(values[3:]).item() == 0


def test_generate_dataset_structured_returns_metadata() -> None:
    dataset, metadata = generate_dataset(
        n_samples=128,
        input_dim=12,
        function_class="teacher_relu",
        noise_std=0.01,
        seed=7,
        data_mode="structured",
        input_rank=5,
        input_spectrum="spiked",
        input_spectrum_decay=2.0,
        teacher_rank=4,
        teacher_spectrum="power_law",
        teacher_spectrum_decay=1.2,
        teacher_hidden_dim=6,
        teacher_activation="relu",
        target_scale=1.0,
    )
    features, targets = dataset.tensors

    assert features.shape == (128, 12)
    assert targets.shape == (128, 1)
    assert metadata.data_mode == "structured"
    assert metadata.input_rank == 5
    assert metadata.teacher_rank == 4
    assert metadata.teacher_hidden_dim == 6
    assert metadata.teacher_activation == "relu"
    assert math.isfinite(metadata.input_effective_rank)
    assert math.isfinite(metadata.teacher_effective_rank)
    assert metadata.target_std > 0.05


def test_generate_dataset_legacy_keeps_existing_path() -> None:
    dataset, metadata = generate_dataset(
        n_samples=64,
        input_dim=8,
        function_class="sine",
        noise_std=0.0,
        seed=11,
    )
    features, targets = dataset.tensors

    assert features.shape == (64, 8)
    assert targets.shape == (64, 1)
    assert metadata.data_mode == "legacy"
    assert metadata.function_class == "sine"


def test_build_teacher_first_layer_identity_is_diagonal() -> None:
    weight, singular_values = build_teacher_first_layer(
        input_dim=5,
        hidden_dim=5,
        teacher_rank=5,
        teacher_spectrum="identity",
        teacher_spectrum_decay=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.allclose(weight, torch.diag(torch.diag(weight)))
    assert torch.allclose(torch.diag(weight), singular_values)
