import jax.numpy as jnp
from equinox import filter_custom_vjp
from jaxtyping import Array

DEFAULT_THRESHOLD = 0.0
DEFAULT_SCALE = 100.0


@filter_custom_vjp
def fast_sigmoid(
    vjp_arg: Array, threshold: float = DEFAULT_THRESHOLD, scale: float = DEFAULT_SCALE
) -> Array:
    return (vjp_arg - threshold > 0).astype(vjp_arg.dtype)


@fast_sigmoid.def_fwd
def fast_sigmoid_fwd(
    perturbed: bool,
    vjp_arg: Array,
    threshold: float = DEFAULT_THRESHOLD,
    scale: float = DEFAULT_SCALE,
):
    # primal output of fwd and residuals/context for bwd
    return fast_sigmoid(vjp_arg, threshold, scale), ()


@fast_sigmoid.def_bwd
def fast_sigmoid_bwd(
    residuals: Array,
    grad_obj: Array,
    perturbed: bool,
    vjp_arg: Array,
    threshold: float = DEFAULT_THRESHOLD,
    scale: float = DEFAULT_SCALE,
) -> Array:
    # treated as if sigmoid in backward pass
    surr_grad = grad_obj / (scale * jnp.abs(vjp_arg - threshold) + 1) ** 2

    return surr_grad
