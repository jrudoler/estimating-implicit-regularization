import torch
from torch.func import vmap, vjp, jvp  # noqa: E402


def compute_empirical_ntk(
    func: callable,
    params: dict,
    x1: torch.Tensor,
    x2: torch.Tensor,
    compute: str = "full",
):
    """
    Using vector jacobian products: This method reformulates NTK as a
    stack of NTK-vector products applied to columns of an identity matrix
    of size (output_dim, output_dim). This is a more efficient way to compute
    the NTK than computing the full Jacobian and then contracting it.

    Args:
        func (callable): The function representing the neural network.
        params (dict): The parameters of the neural network.
        x1 (torch.Tensor): The first set of input data points.
        x2 (torch.Tensor): The second set of input data points.
        compute (str): The type of NTK computation to perform. Options are 'full', 'trace', or 'diagonal'.

    Returns:
        torch.Tensor: The computed NTK matrix based on the specified computation type.
            Shape is (x1.shape[0], x2.shape[0], output_dim, output_dim).
    """

    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes ``vec @ J(x2).T``
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes ``J(X1) @ vjps``
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(
            output.numel(), dtype=output.dtype, device=output.device
        ).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to ``compute_empirical_ntk`` are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the ``vmaps`` here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

    if compute == "full":
        return result
    if compute == "trace":
        return torch.einsum("NMKK->NM", result)
    if compute == "diagonal":
        return torch.einsum("NMKK->NMK", result)
