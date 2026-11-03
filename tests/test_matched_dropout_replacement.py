from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from core.dropout_convergence import train_epoch as original_train_epoch

spec = importlib.util.spec_from_file_location('matched', Path(__file__).resolve().parents[1] / 'analysis/matched_dropout_replacement/run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_zero_penalty_replays_original_epoch() -> None:
    torch.manual_seed(42)
    a = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Dropout(0.), nn.Linear(5, 2))
    b = copy.deepcopy(a)
    data = m.TensorBatches(torch.randn(11, 3), torch.arange(11) % 2, 4)
    oa = torch.optim.SGD(a.parameters(), lr=.05, momentum=.9)
    ob = torch.optim.SGD(b.parameters(), lr=.05, momentum=.9)
    torch.manual_seed(99)
    first = original_train_epoch(a, data, oa)
    torch.manual_seed(99)
    second = m.train_epoch(b, data, ob, 0.)
    assert first == second
    assert all(torch.equal(x, y) for x,y in zip(a.parameters(),b.parameters()))


def test_ridge_gradient_includes_bias_and_preserves_ce_stopping() -> None:
    torch.manual_seed(7)
    a = nn.Linear(3, 2)
    b = copy.deepcopy(a)
    data = m.TensorBatches(torch.randn(7, 3), torch.arange(7) % 2, 7)
    oa = torch.optim.SGD(a.parameters(), lr=.05, momentum=.9)
    ob = torch.optim.SGD(b.parameters(), lr=.05, momentum=.9)
    torch.manual_seed(19)
    order = torch.randperm(7)
    ce = F.cross_entropy(a(data.x[order]), data.y[order])
    loss = ce + .02 * sum(p.square().sum() for p in a.parameters())
    loss.backward()
    oa.step()
    torch.manual_seed(19)
    measured = m.train_epoch(b, data, ob, .02)
    assert abs(measured - float(ce.detach())) < 1e-7
    for x,y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)


def test_prediction_agreement_distinct_from_accuracy() -> None:
    target = torch.tensor([[2.,0.], [0.,2.], [2.,0.]])
    replacement = target[[1,0,2]]
    metrics = m.compare_logits(replacement,target,torch.tensor([0,1,0]))
    assert metrics['prediction_disagreement'] == 2/3
    assert metrics['target_accuracy'] == 1.
    assert metrics['accuracy'] == 1/3


def test_summary_rejects_partial_experiment(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location('matched_analysis', Path(__file__).resolve().parents[1] / 'analysis/matched_dropout_replacement/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import pytest
    with pytest.raises(ValueError, match='Incomplete'):
        module.summarize([], tmp_path)


def test_paired_difference_zero_and_constant() -> None:
    spec = importlib.util.spec_from_file_location('matched_analysis', Path(__file__).resolve().parents[1] / 'analysis/matched_dropout_replacement/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.paired_difference([0.] * 10)['mean_bootstrap_ci95'] == [0.,0.]
    values = module.paired_difference([.01] * 10)
    assert abs(values['mean']-.01) < 1e-15
    assert all(abs(x-.01) < 1e-15 for x in values['mean_bootstrap_ci95'])
