A SciPy and PyTorch implementation of the proximal operator $\mathrm{prox}_{\eta f}(w)$ of functions of the form:
```math
f(w;\theta, \phi, b, \alpha) = \exp(\langle \theta, w \rangle + b) + \langle \phi, w \rangle + \frac{\alpha}{2} \| w \|_2^2
```
This repository contains two modules, `exp_prox.np` and `exp_prox.torch`, both contain a function with the following signature:
```python
def prox_op(w: Array, eta: Array, theta: Array, phi: Array, b: Union[Float, Array], alpha: Union[Float, Array]) -> Array
```
The functions support mini-batches, by treating all but the last dimension as mini-batch dimensions.

Functions of the above family appear, for example, as _regularized_ losses of Poisson regression. To that end, we also have a utility function specifically for incremental Poisson regression. For example, the snippet below implements incremental proximal-point algorithm for Poisson regression:
```python
from exp_prox import poisson_params
from exp_proc.np import prox_op

step_size = 1e-3
reg_coef = 1e-5
w = np.zeros(num_features) # the learned model weights
for X, y in data_set:
    w = prox_op(w, *poisson_params(step_size, X, y, reg_coef))
```
