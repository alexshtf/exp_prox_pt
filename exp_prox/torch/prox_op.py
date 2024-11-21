import torch
from numpy.typing import ArrayLike
from typing import Union
import numbers
from .wrightomega import wrightomega


def is_scalar(x):
    if isinstance(x, numbers.Number):
        return True
    elif torch.asarray(x).ndim == 0:
        return True

    return False

def prox_op(w: ArrayLike,
            eta: Union[ArrayLike, float],
            theta: ArrayLike,
            phi: ArrayLike,
            b: Union[ArrayLike, float],
            alpha: Union[ArrayLike, float]):
    w = torch.as_tensor(w)
    eta = torch.as_tensor(eta)
    theta = torch.as_tensor(theta)
    phi = torch.as_tensor(phi)
    b = torch.as_tensor(b)
    alpha = torch.as_tensor(alpha)

    # broadcast input arguments expected to have one-less dimension than w, theta, and phi if they are not scalars
    if eta.ndim > 0:
        eta = torch.asarray(eta)[..., torch.newaxis]
    if b.ndim > 0:
        b = torch.asarray(b)[..., torch.newaxis]
    if alpha.ndim > 0:
        alpha = torch.asarray(alpha)[..., torch.newaxis]

    # compute formula parts
    common_denom = (1 + eta * alpha)
    gamma = eta * torch.sum(torch.square(theta), dim=-1, keepdim=True) / common_denom
    delta = torch.sum(theta * (w - eta * phi), dim=-1, keepdim=True) / common_denom + b

    # solve q'(s) = 0
    s = wrightomega(delta + torch.log(gamma)) / gamma

    # compute the result
    return (w - eta * s * theta - eta * phi) / common_denom