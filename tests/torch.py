import cvxpy as cp
import numpy as np
import pytest

from exp_prox.torch import prox_op

def test_correctness():
    ''' compare to CVXPY results '''
    m = 10
    n = 5

    # define function parameters
    thetas = np.geomspace(1, 100, m * n).reshape(m, n)
    phis = np.geomspace(1, 100, m * n).reshape(m, n)
    bs = np.geomspace(10, 100, m)
    alphas = np.geomspace(0.1, 10, m)

    # define prox operator argument and step-size
    ws = np.linspace(-1, 1, m * n).reshape(m, n)
    etas = np.geomspace(0.1, 1, m)

    # compute using our prox_op
    w_next = prox_op(ws, etas, thetas, phis, bs, alphas)
    w_next = w_next.numpy()

    # compute using CVXPY
    w_next_cp = []
    for row in range(m):
        u = cp.Variable(n)
        objective = (
            cp.exp(cp.vdot(u, thetas[row]) + bs[row]) +
            cp.vdot(u, phis[row]) +
            alphas[row] * cp.sum_squares(u) / 2 +
            cp.sum_squares(ws[row] - u) / (2 * etas[row])
        )
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve()
        w_next_cp.append(u.value)
    w_next_cp = np.stack(w_next_cp)

    diff = np.max(np.abs(w_next - w_next_cp))
    assert float(diff) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__])