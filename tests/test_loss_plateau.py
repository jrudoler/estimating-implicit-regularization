from copy import deepcopy

import pytest

from core.loss_plateau import LossPlateau, coefficient_stability


def test_plateau_reduces_lr_then_waits_full_patience() -> None:
    rule = LossPlateau(window=2, patience=6, lr_patience=4)
    lr, actions = .05, []
    for epoch in range(1, 25):
        record = rule.update(1., lr)
        if record:
            lr = record["next_lr"]
            if record["action"] != "continue":
                actions.append((epoch, record["action"]))
    assert actions[:3] == [(6, "reduce_lr"), (12, "reduce_lr"), (20, "stop")]
    assert lr == pytest.approx(.0005)


def test_averages_all_epochs_and_uses_relative_improvement() -> None:
    rule = LossPlateau(window=2, patience=6, lr_patience=4)
    assert rule.update(.5, .0005) is None
    assert rule.update(1.5, .0005)["mean_train_loss"] == 1.
    rule.update(.9995, .0005)
    assert rule.update(.9995, .0005)["bad_epochs"] == 2
    rule.update(.998, .0005)
    assert rule.update(.998, .0005)["bad_epochs"] == 0


def test_resume_midblock_matches_uninterrupted() -> None:
    rule = LossPlateau(window=2, patience=6, lr_patience=4)
    rule.update(1., .05)
    resumed = LossPlateau(**deepcopy(rule.state_dict()))
    for loss in [1., .99, 1., .98, .97, .97, .97]:
        assert rule.update(loss, .05) == resumed.update(loss, .05)
    with pytest.raises(ValueError):
        rule.update(float("nan"), .05)


def test_absolute_tolerance_ignores_tiny_relative_improvements() -> None:
    rule = LossPlateau(window=1, patience=4, lr_patience=2,
                       relative_improvement=0., absolute_improvement=1e-7)
    rule.update(1e-6, .0005)
    assert rule.update(.95e-6, .0005)["bad_epochs"] == 1
    assert rule.update(.91e-6, .0005)["bad_epochs"] == 2
    assert rule.update(.89e-6, .0005)["bad_epochs"] == 0
    for _ in range(3):
        assert rule.update(.85e-6, .0005)["action"] == "continue"
    assert rule.update(.85e-6, .0005)["action"] == "stop"


def test_stability_does_not_hide_large_relative_changes_outside_zero_band() -> None:
    assert coefficient_stability(1e-10, 2e-10)["both_practically_zero"]
    assert not coefficient_stability(1.1e-7, .5e-7)["passes"]
    assert coefficient_stability(1e-5, 1.01e-5)["passes"]
    assert not coefficient_stability(1e-5, -1e-5)["passes"]
