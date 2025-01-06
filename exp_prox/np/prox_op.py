import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from scipy.special import wrightomega

def prox_op(w: ArrayLike,
            eta: Union[ArrayLike, float],
            theta: ArrayLike,
            phi: ArrayLike,
            b: Union[ArrayLike, float],
            alpha: Union[ArrayLike, float],
            return_dual=False):
    w = np.asarray(w)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # broadcast input arguments expected to have one-less dimension than w, theta, and phi if they are not scalars
    if not np.isscalar(eta):
        eta = np.asarray(eta)[..., np.newaxis]
    if not np.isscalar(b):
        b = np.asarray(b)[..., np.newaxis]
    if not np.isscalar(alpha):
        alpha = np.asarray(alpha)[..., np.newaxis]

    # compute formula parts
    common_denom = (1 + eta * alpha)
    gamma = eta * np.sum(np.square(theta), axis=-1, keepdims=True) / common_denom
    delta = np.sum(theta * (w - eta * phi), axis=-1, keepdims=True) / common_denom + b

    # solve q'(s) = 0
    s = wrightomega(delta + np.log(gamma)) / gamma

    # compute the result
    solution = (w - eta * s * theta - eta * phi) / common_denom
    if not return_dual:
        return solution
    else:
        return solution, s
