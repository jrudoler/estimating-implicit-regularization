from copy import deepcopy
from pathlib import Path

import pytest
import torch

from core.dropout_continuation import restore_rng, scheduled_lr, validate_import
from core.dropout_convergence import train_epoch
from core.scalar_estimation import file_sha256, model_sha256
from types import SimpleNamespace


def test_sustained_schedule_and_boundary_restarts() -> None:
    kwargs = dict(initial=.05, milestones=[40000, 50000, 55000, 58000], factor=.1)
    assert scheduled_lr(39999, **kwargs) == .05
    assert scheduled_lr(40000, **kwargs) == pytest.approx(.005)
    assert scheduled_lr(59999, **kwargs) == pytest.approx(.000005)
    with pytest.raises(ValueError):
        scheduled_lr(0, initial=.05, milestones=[500, 100], factor=.1)


def test_import_preserves_original_baseline_and_rejects_wrong_source(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "epoch_04000.ckpt"
    signature = {"config": {"seed": 123}, "protocol": {"max_epochs": 4000}}
    payload = {"epoch": 4118, "additional_epochs": 4000, "signature": signature, "baseline_norm": .012}
    torch.save(payload, checkpoint)
    expected = {"epoch": 4118, "checkpoint_sha256": file_sha256(checkpoint), "model_sha256": model_sha256(model)}
    source = {**signature, "status": "budget_exhausted", "baseline_norm": .012,
              "trajectory": [{"additional_epochs": 4000, "provenance": expected}]}
    assert validate_import(source, payload, checkpoint, model) == expected
    for key, value in (("baseline_norm", .002), ("epoch", 4117), ("additional_epochs", 3999)):
        wrong = {**payload, key: value}
        with pytest.raises(ValueError):
            validate_import(source, wrong, checkpoint, model)
    wrong_source = deepcopy(source)
    wrong_source["protocol"]["max_epochs"] = 60000
    with pytest.raises(ValueError, match="signature"):
        validate_import(wrong_source, payload, checkpoint, model)
    with torch.no_grad():
        model.weight.add_(.1)
    with pytest.raises(ValueError, match="model hash"):
        validate_import(source, payload, checkpoint, model)


def test_imported_rng_and_momentum_replay_across_lr_boundary() -> None:
    torch.manual_seed(22)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(),
                                torch.nn.Dropout(.3), torch.nn.Linear(5, 2))
    data = SimpleNamespace(x=torch.randn(17, 3), y=torch.arange(17) % 2, batch_size=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9)
    train_epoch(model, data, optimizer)
    payload = {"cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    weights, momentum = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())

    def next_stage() -> float:
        optimizer.param_groups[0]["lr"] = scheduled_lr(40000, initial=.05, milestones=[40000], factor=.1)
        return train_epoch(model, data, optimizer)

    first_loss = next_stage()
    first_weights = deepcopy(model.state_dict())
    model.load_state_dict(weights)
    optimizer.load_state_dict(momentum)
    torch.rand(100)  # Loading a model or other setup must not perturb training.
    restore_rng(payload)
    assert next_stage() == first_loss
    assert all(torch.equal(value, first_weights[key]) for key, value in model.state_dict().items())
