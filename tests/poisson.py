from exp_prox import poisson_params
from exp_prox.np import prox_op
import numpy as np
import cvxpy as cp

def test_poisson_params():
    w = np.zeros(3)
    features = np.arange(12).reshape(4, 3)
    labels = np.array([1, 3, 2, 4])
    step_size = 20
    reg_coef = 0.1

    # compute directly
    w_next = prox_op(w, *poisson_params(step_size, features, labels, reg_coef))

    # compute using cvxpy
    w_next_cp = cp.Variable(features.shape)
    pred_cp = cp.sum(cp.multiply(features, w_next_cp), axis=1)
    losses = cp.exp(pred_cp) - cp.multiply(labels, pred_cp)
    regs = cp.sum(cp.square(w_next_cp), axis=1)
    dists = cp.sum(cp.square(w_next_cp - w[np.newaxis, :]), axis=1)
    objective = cp.sum(losses + reg_coef * regs / 2 + dists / (2 * step_size))
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve()

    assert w_next.shape == features.shape
    assert np.allclose(w_next, w_next_cp.value, atol=1e-5)
