A SciPy and PyTorch implementation of the proximal operator $\mathrm{prox}_{\eta f}(w)$ of functions of the form:
```math
f(w;\theta, \phi, b, alpha) = \exp(\langle \theta, w \rangle + b) + \langle \phi, w \rangle + \frac{\alpha}{2} \| w \|_2^2
```
Such functions appear, for example, as _regularized_ losses of Poisson regression.

This repository contains two modules, `exp_prox.np` and `exp_prox.torch`, both contain a function with the following signature:
```python
def prox_op(w: Array, eta: Array, theta: Array, phi: Array, b: Union[Float, Array], alpha: Union[Float, Array]) -> Array
```
Supports mini-batching by treating only the last dimension of 'w' as the argument, whereas the previous dimensions are mini-batch dimensions.
