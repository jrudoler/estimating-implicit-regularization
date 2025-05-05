import jax
import jax.numpy as jnp
from flax.experimental import nnx
from typing import Sequence

PRNGKey = jax.Array


class NoisyMLP(nnx.Module):
    """A simple Multi-Layer Perceptron with dropout and Gaussian noise."""

    def __init__(
        self,
        features: Sequence[int],
        dropout_rate: float,
        noise_std: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.layers = []
        for i, feat in enumerate(features):
            self.layers.append(
                nnx.Linear(features[i - 1] if i > 0 else None, feat, rngs=rngs)
            )  # Input size inferred later
            if i < len(features) - 1:  # No activation/dropout on the last layer
                self.layers.append(
                    nnx.Dropout(rate=self.dropout_rate, deterministic=False, rngs=rngs)
                )  # Need to handle deterministic mode during eval

    def __call__(
        self, x: jax.Array, *, rngs: nnx.Rngs | None = None, deterministic: bool = False
    ) -> jax.Array:
        """Applies the MLP layers with noise and dropout."""
        # NNX modules need explicit RNG handling for dropout/initialization
        dropout_rng = rngs.dropout() if rngs is not None else None

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nnx.Linear):
                x = layer(x)
                if (
                    i < len(self.layers) - 1
                ):  # Apply activation after linear if not the last layer
                    x = nnx.relu(x)
            elif isinstance(layer, nnx.Dropout):
                # Pass deterministic flag to dropout
                x = layer(
                    x,
                    deterministic=deterministic,
                    rngs=nnx.Rngs(dropout=dropout_rng)
                    if dropout_rng is not None
                    else None,
                )

        # Add Gaussian noise to the output
        if self.noise_std > 0 and not deterministic:
            noise_rng = (
                rngs.noise() if rngs is not None else None
            )  # Requires a 'noise' RNG stream
            if noise_rng is None:
                raise ValueError(
                    "RNG for noise must be provided when noise_std > 0 and not deterministic."
                )
            noise = jax.random.normal(noise_rng, x.shape) * self.noise_std
            x = x + noise
        return x


class LinearNetwork(nnx.Module):
    """A simple feedforward network with linear layers and ReLU activations."""

    def __init__(self, features: Sequence[int], *, rngs: nnx.Rngs):
        self.layers = []
        for i, feat in enumerate(features):
            # Input size needs to be known or inferred during first call/init
            self.layers.append(
                nnx.Linear(features[i - 1] if i > 0 else None, feat, rngs=rngs)
            )
            if i < len(features) - 1:  # No activation on the last layer
                self.layers.append(nnx.relu)  # Use jax.nn.relu directly

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


class LinearRegression(nnx.Module):
    """A simple linear regression model (single linear layer)."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# Kernel Regression is non-parametric, so it doesn't fit the nnx.Module pattern well.
# It's better implemented as a function that takes data and computes predictions.


def kernel_regression_predict(
    train_x: jax.Array,
    train_y: jax.Array,
    test_x: jax.Array,
    kernel_fn,
    **kernel_params,
) -> jax.Array:
    """Performs kernel regression prediction.

    Args:
        train_x: Training input data (n_samples, n_features).
        train_y: Training target data (n_samples, n_outputs).
        test_x: Test input data (m_samples, n_features).
        kernel_fn: A function that computes the kernel matrix K(X1, X2).
                   Example: lambda x1, x2, sigma: jnp.exp(-jnp.sum((x1[:, None] - x2[None, :])**2, axis=-1) / (2 * sigma**2))
        **kernel_params: Parameters for the kernel function (e.g., sigma for RBF).

    Returns:
        Predictions for test_x (m_samples, n_outputs).
    """
    # Compute kernel matrix between test and train data
    k_test_train = kernel_fn(test_x, train_x, **kernel_params)

    # Compute kernel matrix for training data (with regularization)
    k_train_train = kernel_fn(train_x, train_x, **kernel_params)
    # Add small epsilon for numerical stability (ridge regression)
    k_train_train_reg = k_train_train + jnp.eye(train_x.shape[0]) * 1e-6

    # Solve for weights (alpha = (K_train_train + lambda*I)^-1 * y_train)
    # Use jax.scipy.linalg.solve for better stability/performance than inv
    alpha = jax.scipy.linalg.solve(k_train_train_reg, train_y, assume_a="pos")

    # Make predictions (y_pred = K_test_train * alpha)
    predictions = jnp.dot(k_test_train, alpha)
    return predictions


# Example RBF kernel function (can be defined elsewhere or passed in)
def rbf_kernel(x1, x2, sigma):
    """Radial Basis Function (RBF) kernel."""
    # Ensure inputs are at least 2D
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    # Calculate squared distances (efficiently)
    sq_dist = (
        jnp.sum(x1**2, axis=1)[:, None]
        + jnp.sum(x2**2, axis=1)[None, :]
        - 2 * jnp.dot(x1, x2.T)
    )
    return jnp.exp(-sq_dist / (2 * sigma**2))


# TODO: Refine NNX initialization (input shapes)
# TODO: Ensure RNG handling is correct for all modules, especially during initialization vs. calls.
