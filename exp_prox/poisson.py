from numpy.typing import ArrayLike
from typing import Union

def poisson_params(
        step_size: Union[ArrayLike, float],
        features: ArrayLike,
        label: Union[ArrayLike, float],
        reg_coef: Union[ArrayLike, float]=0.):
    theta = features
    phi = -label * features
    return step_size, theta, phi, 0, reg_coef
