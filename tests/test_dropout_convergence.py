from copy import deepcopy
from types import SimpleNamespace

import torch

from core.dropout_convergence import assess_convergence, continuation_lr, train_epoch


def small_control():
    return {"endpoint": {"scale_star": 2e-5},
            "direct_signed_lambda_mc_ci95": [1.99e-5, 2.01e-5],
            "clean_gradient_norm": .02,
            "train_gradient_norm_sq_ci95": [-1e-8, 1e-8],
            "train_gradient_norm_sq_unbiased": 0.,
            "signed_coefficient_contamination": 0.}


def test_requires_small_norm_upper_bound_not_zero_overlap():
    control = small_control()
    assert assess_convergence(control, baseline_norm=.01, dropout=.3)["passes"]
    control["train_gradient_norm_sq_ci95"] = [-1., 1.]
    assert not assess_convergence(control, baseline_norm=.01, dropout=.3)["passes"]
    control["train_gradient_norm_sq_ci95"] = [-2., -1.]
    assert not assess_convergence(control, baseline_norm=.01, dropout=.3)["passes"]


def test_coefficient_bias_uncertainty_must_fit_inside_tolerance():
    control = small_control()
    control["direct_signed_lambda_mc_ci95"] = [0., 4e-5]
    result = assess_convergence(control, baseline_norm=.01, dropout=.3)
    assert result["norm_pass"]
    assert not result["coefficient_pass"]


def test_zero_dropout_requires_small_absolute_endpoint_and_gradient():
    control = small_control()
    control["endpoint"]["scale_star"] = 5e-8
    control["direct_signed_lambda_mc_ci95"] = [0., 0.]
    control["clean_gradient_norm"] = 5e-5
    control["train_gradient_norm_sq_ci95"] = [2.5e-9, 2.5e-9]
    assert assess_convergence(control, baseline_norm=.01, dropout=0)["passes"]
    control["endpoint"]["scale_star"] = 2e-5
    assert not assess_convergence(control, baseline_norm=.01, dropout=0)["passes"]


def test_continuation_changes_weights_and_restored_rng_optimizer_replay():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(),
                                torch.nn.Dropout(.3), torch.nn.Linear(6, 2))
    data = SimpleNamespace(x=torch.randn(11, 3), y=torch.arange(11) % 2, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    train_epoch(model, data, optimizer)
    model_state, opt_state, rng = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), torch.get_rng_state()
    loss1 = train_epoch(model, data, optimizer)
    expected = deepcopy(model.state_dict())
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    torch.set_rng_state(rng)
    loss2 = train_epoch(model, data, optimizer)
    assert loss1 == loss2
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())
    assert any(not torch.equal(value, model_state[key]) for key, value in model.state_dict().items())
    assert continuation_lr(999) == .01
    assert continuation_lr(1000) == .002
