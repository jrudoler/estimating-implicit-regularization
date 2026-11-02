import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.scalar_estimation import (scalar_projection, full_batch_loss_grad,
                                    dropout_control, model_sha256)
from core.dropout_grid import EXPECTED, load_grid


def test_exact_projection_and_nonnegative_boundary():
    basis = torch.tensor([2., -3., 4.], dtype=torch.float64)
    target = .00004 * basis
    fit = scalar_projection(basis, target)
    reference = torch.linalg.lstsq(basis[:, None], target[:, None]).solution.item()
    assert fit["scale_star_clamped"] == pytest.approx(reference, abs=1e-15)
    assert fit["residual_ratio_clamped"] < 1e-14
    negative = scalar_projection(basis, -target)
    assert negative["scale_star"] < 0
    assert negative["scale_star_clamped"] == 0
    assert negative["residual_ratio_clamped"] == 1
    assert negative["projection_r2_clamped"] == 0


def test_degenerate_targets():
    fit = scalar_projection(torch.ones(3), torch.zeros(3))
    assert fit["scale_star_clamped"] == 0
    assert fit["grad_cosine"] is None
    assert fit["residual_ratio_clamped"] is None
    assert scalar_projection(torch.zeros(3), torch.ones(3))["status"] == "zero_basis"
    with pytest.raises(ValueError, match="Nonfinite"):
        scalar_projection(torch.ones(3), torch.full((3,), float("nan")))


def test_full_gradient_unequal_batches_and_preservation():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.5), nn.Linear(4, 2)).double().train()
    model[1].eval()  # Deliberately mixed modes must be restored exactly.
    x, y = torch.randn(11, 3, dtype=torch.float64), torch.arange(11) % 2
    data = TensorDataset(x, y)
    before = model_sha256(model)
    modes = [m.training for m in model.modules()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    first = full_batch_loss_grad(model, DataLoader(data, batch_size=4))
    second = full_batch_loss_grad(model, DataLoader(data, batch_size=11))
    assert torch.allclose(first, second, atol=1e-14, rtol=1e-12)
    assert [m.training for m in model.modules()] == modes
    assert before == model_sha256(model)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    with pytest.raises(ValueError, match="empty"):
        full_batch_loss_grad(model, [])


def test_zero_dropout_control_and_state_preservation():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.), nn.Linear(4, 2)).double()
    loader = DataLoader(TensorDataset(torch.randn(9, 3, dtype=torch.float64), torch.arange(9) % 2), batch_size=4)
    # The clean pass may construct a DataLoader iterator; fork at caller level
    # when preserving RNG across the complete diagnostic is required.
    before = model_sha256(model)
    result = dropout_control(model, loader, draws=4, bootstrap=100)
    assert result["direct_dropout"]["scale_star"] == 0
    assert result["direct_signed_lambda_mc_ci95"] == [0., 0.]
    assert result["train_gradient_norm_sq_unbiased"] == pytest.approx(result["clean_gradient_norm"]**2)
    assert before == model_sha256(model)


def test_negative_unbiased_norm_is_not_reported_as_convergence(monkeypatch):
    import core.scalar_estimation as module
    model = nn.Linear(1, 1, bias=False).double()
    gradients = iter([torch.tensor([1.]), torch.tensor([1.]), torch.tensor([-1.]),
                      torch.tensor([1.]), torch.tensor([-1.])])
    monkeypatch.setattr(module, "full_batch_loss_grad", lambda *args, **kwargs: next(gradients).double())
    control = module.dropout_control(model, [], draws=4, bootstrap=100)
    assert control["train_gradient_norm_sq_unbiased"] < 0
    assert "train_gradient_norm_unbiased" not in control
    assert "not proof of convergence" in control["interpretation"]


def test_lightning_keeps_predictor_in_eval():
    from lightning import Trainer, Callback
    from core.bias import RidgeBias
    from core.estimators import BiasWithCrossEntropyScheduled
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.5), nn.Linear(4, 2))
    estimator = BiasWithCrossEntropyScheduled(predictive_model=model, bias_model=RidgeBias())
    before = model_sha256(model)
    class Check(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            assert all(not m.training for m in model.modules())
    estimator.train()
    assert not model.training
    loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.arange(8) % 2), batch_size=4)
    Trainer(max_steps=2, accelerator="cpu", logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, callbacks=[Check()]).fit(estimator, train_dataloaders=loader)
    assert before == model_sha256(model)


def test_grid_rejects_duplicate_missing_and_stale_receipt(tmp_path):
    from analysis.plot_closed_form_dropout_grid.run import load_results
    from core.scalar_estimation import file_sha256
    sources = []
    for i, (d, w, p, s) in enumerate(sorted(EXPECTED)):
        path = tmp_path / f"cell_{i}.json"
        record = {"config": dict(depth=d, width=w, dropout=p, seed=s),
                  "closed_form": {"ridge": scalar_projection(torch.ones(3), torch.ones(3))}}
        path.write_text(json.dumps(record))
        sources.append({"cell": [d, w, p, s], "sha256": file_sha256(path)})
    receipt = tmp_path.parent / "receipt.json"
    receipt.write_text(json.dumps({"sources": sources}))
    assert len(load_results(tmp_path, receipt)) == 480
    first = tmp_path / "cell_0.json"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(first.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        load_grid(tmp_path)
    duplicate.unlink()
    first.write_text(first.read_text() + "\n")
    with pytest.raises(ValueError, match="Stale"):
        load_results(tmp_path, receipt)
    first.unlink()
    with pytest.raises(ValueError, match="Missing"):
        load_grid(tmp_path)


@pytest.mark.parametrize("method", ["auto", "legacy_iterative"])
def test_production_default_uses_checkpoint_weights(tmp_path, monkeypatch, method):
    from analysis.dropout_bias_estimation import run
    from core.models import DeepReLUClassifier
    model = DeepReLUClassifier(input_dim=3, num_classes=2, depth=2, width=4,
                               dropout=.3, batchnorm=False, l2_lambda=0., lr=.05, momentum=.9)
    dataset = TensorDataset(torch.randn(9, 3), torch.arange(9) % 2)
    class Data:
        def __init__(self, **kwargs):
            self._train_dataset = dataset
        def prepare_data(self):
            pass
        def setup(self, stage):
            pass
        def train_dataloader(self):
            return DataLoader(dataset, batch_size=4)
    monkeypatch.setattr(run, "MNISTLightningDataModule", Data)
    monkeypatch.setattr(run.DeepReLUClassifier, "load_from_checkpoint", lambda _: model)
    class LegacyTrainer:
        callback_metrics = {}
        def __init__(self, **kwargs):
            assert method == "legacy_iterative", "Closed form must not create an estimator Trainer"
        def fit(self, estimator, train_dataloaders):
            estimator.setup("fit")
    monkeypatch.setattr(run, "Trainer", LegacyTrainer)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"epoch": 7}, checkpoint)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 42, "batch_size": 4, "num_data_workers": 0}))
    out = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["run", "--checkpoint-path", str(checkpoint), "--config-path", str(config),
                                     "--disable-wandb", "--output", str(out), "--estimator", method])
    before = model_sha256(model)
    run.main()
    result = json.loads(out.read_text())
    assert result["estimator"]["method"] == ("closed_form" if method == "auto" else method)
    assert result["provenance"]["epoch"] == 7
    assert result["provenance"]["model_sha256"] == before
    assert result["config"]["depth"] == 2  # Actual loaded config, not defaults.
    assert result["bias_params"]["estimated_ridge"] >= 0
    assert before == model_sha256(model)
    if method == "legacy_iterative":
        assert result["adequacy"]["scope"] == "full_training_dataset"
        assert result["adequacy"]["projection_r2"] == pytest.approx(1 - result["adequacy"]["residual_ratio"]**2)


def test_optimizer_audit_reference_and_matched_updates():
    from analysis.dropout_checkpoint_audit.run import optimizer_audit
    torch.manual_seed(5)
    model = nn.Linear(3, 2).double()
    loader = DataLoader(TensorDataset(torch.randn(9, 3, dtype=torch.float64), torch.arange(9) % 2), batch_size=4)
    result = optimizer_audit(model, loader, epochs=2)
    assert result["closed_form"]["estimated_ridge"] == pytest.approx(result["reference_nonnegative_lstsq"], abs=1e-14)
    assert {v["updates"] for v in result["variants"].values()} == {6}
    assert all(v["full_dataset_sse_gap"] >= -1e-14 for v in result["variants"].values())


def test_control_summary_rejects_inconsistent_diagnostics():
    from copy import deepcopy
    from analysis.dropout_checkpoint_audit.analyze import validate_record
    basis = torch.ones(3, dtype=torch.float64)
    record = {
        "control": {"mask_draws": 32, "direct_signed_lambda_draws": [.1] * 32,
                    "endpoint": scalar_projection(basis, .2 * basis),
                    "direct_dropout": scalar_projection(basis, .1 * basis),
                    "signed_coefficient_contamination": -.1},
        "provenance": {"checkpoint_sha256": "test", "dataset": {"split_indices_sha256": "test"}},
    }
    validate_record(record)
    bad = deepcopy(record)
    bad["control"]["direct_dropout"]["projection_r2_clamped"] = .5
    with pytest.raises(ValueError, match="diagnostics"):
        validate_record(bad)
    bad = deepcopy(record)
    bad["control"]["signed_coefficient_contamination"] = .1
    with pytest.raises(ValueError, match="decomposition"):
        validate_record(bad)


def test_checkpoint_diagnostics_never_start_missing_training(tmp_path, monkeypatch):
    from analysis.dropout_checkpoint_audit import run
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"config": {"depth": 3, "width": 256, "dropout": .5, "seed": 42},
                                "checkpoints": [], "endpoint_kind": "explicit_replacement_replay",
                                "replay_record": str(tmp_path / "missing.json")}))
    monkeypatch.setattr("sys.argv", ["run", "--spec", str(spec), "--output", str(tmp_path / "out.json"),
                                     "--data-root", str(tmp_path), "--device", "cpu"])
    with pytest.raises(FileNotFoundError, match="workflow rule first"):
        run.main()
