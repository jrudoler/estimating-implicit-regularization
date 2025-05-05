import jax
import jax.numpy as jnp
import optax
from typing import Any, Tuple, Optional
from optax._src import base  # for the ScalarOrSchedule alias

"""
optimization.py

Custom Optax-based optimizers for kernel gradient descent.

This module provides:
- `_kernel_preconditioner`: a transformation that applies a precomputed inverse kernel matrix to gradients.
- `kernel_sgd`: an SGD optimizer factory that chains the kernel preconditioner with a standard learning rate scaling.
"""


def _kernel_preconditioner(
    kernel_inv: jnp.ndarray,
) -> optax.GradientTransformation:
    """Multiply flattened gradients by the kernel inverse (stateless)."""

    def init_fn(params: Any) -> optax.EmptyState:
        # signature matches Optax: accept params, then ignore them
        del params
        return optax.EmptyState()

    def update_fn(
        updates: Any,
        state: optax.EmptyState,
        params: Optional[Any] = None,
    ) -> Tuple[Any, optax.EmptyState]:
        # signature matches Optax: accept params, then ignore them
        del params
        # Flatten the gradient pytree into a 1-D vector for kernel multiplication
        flat, unravel = jax.flatten_util.ravel_pytree(updates)
        # Multiply by the inverse kernel to precondition the gradient
        precond_flat = kernel_inv @ flat
        # Unravel the preconditioned flat vector back to the original pytree structure
        return unravel(precond_flat), state

    return optax.GradientTransformation(init_fn, update_fn)


def kernel_sgd(
    kernel_inv: jnp.ndarray,
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    """
    SGD with kernel preconditioning + optional momentum and LR schedule.

    Args:
      kernel_inv:      (n,n) inverse kernel matrix.
      learning_rate:   scalar or schedule fn(step) -> lr_t.
      momentum:        momentum decay (None disables momentum).
      nesterov:        whether to apply Nesterov correction.
      accumulator_dtype: dtype for the momentum accumulator.
    """
    # Precondition
    transforms = [_kernel_preconditioner(kernel_inv)]

    # Optional momentum (trace) block
    if momentum is not None:
        transforms.append(
            optax.trace(
                decay=momentum,
                nesterov=nesterov,
                accumulator_dtype=accumulator_dtype,
            )
        )

    # Schedule‑aware scaling (includes the “–” sign)
    transforms.append(optax.scale_by_learning_rate(learning_rate))

    # Chain them in order
    return optax.chain(*transforms)


# ── Example ────────────────────────────────────────────────────────────────────

# Suppose you have a kernel K of size (n,n):
#    K = compute_kernel(params)
#    K_inv = jnp.linalg.inv(K)
#    η     = 1e-3
#
# Example: compute or load your kernel, invert it, then create and use the kernel-preconditioned SGD
# optimizer = kernel_sgd(K_inv, η)
# opt_state = optimizer.init(params)
# grads     = jax.grad(loss_fn)(params, batch)
# updates, opt_state = optimizer.update(grads, opt_state, params)
# params = optax.apply_updates(params, updates)
#
# Or with nnx.Optimizer:
# optimizer = nnx.Optimizer(kernel_sgd(K_inv, η))
