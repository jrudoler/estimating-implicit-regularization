import torch
import torch.nn as nn
import torch.nn.functional as F


# predictive_loss_grad implementation from the notebook
def predictive_loss_grad(
    predictions: torch.Tensor, targets: torch.Tensor, loss_fn
) -> torch.Tensor:
    loss = loss_fn(predictions, targets)
    per_sample_grad = torch.autograd.grad(
        loss, predictions, retain_graph=True, create_graph=True
    )[0]
    return per_sample_grad


def test_predictive_loss_grad_mse():
    predictions = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    targets = torch.tensor([[1.5], [2.5], [3.5]])
    loss_fn = nn.MSELoss(reduction="sum")
    grad = predictive_loss_grad(predictions, targets, loss_fn)
    expected_grad = -2 * (targets - predictions)
    assert torch.allclose(grad, expected_grad), (
        f"Expected {expected_grad}, but got {grad}"
    )


# def test_predictive_loss_grad_cross_entropy():
#     predictions = torch.tensor([[-0.1, -0.9], [-0.8, -0.2]], requires_grad=True)
#     targets = torch.tensor([1, 0])
#     loss_fn = nn.CrossEntropyLoss(reduction="sum")
#     grad = predictive_loss_grad(predictions, targets, loss_fn)
#     expected_grad = predictions - targets
#     assert torch.allclose(grad, expected_grad), (
#         f"Expected {expected_grad}, but got {grad}"
#     )


def test_predictive_loss_grad_cross_entropy():
    predictions = torch.tensor(
        [[0.1, -0.9], [-0.8, 0.2], [2.1, 0.5]], requires_grad=True
    )
    # values are logits, not probabilities
    targets = torch.tensor([1, 0, 0])
    # targets are class indices, not one-hot encoded
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    grad = predictive_loss_grad(predictions, targets, loss_fn)
    print("Grad shape:", grad.shape)
    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)
    # one hot encode the targets
    one_hot_targets = F.one_hot(targets, num_classes=predictions.shape[1]).float()
    expected_grad = F.softmax(predictions, dim=1) - one_hot_targets
    print("Expected grad shape:", expected_grad.shape)
    assert torch.allclose(grad, expected_grad), (
        f"Expected {expected_grad}, but got {grad}"
    )
