import torch

def wrightomega(x):
    """ Computes the Wright Omega function, based on the algorithm from SciPy:
    https://github.com/scipy/scipy/blob/maintenance/1.14.x/scipy/special/wright.cc#L369

    Example with special values:
    >>>specials = torch.tensor([float('nan'), float('inf'), float('-inf')])
    >>>wrightomega(specials)
    tensor([nan, inf, 0.])

    Example showing accuracy vs SciPy implementation
    >>>import numpy as np
    >>>import scipy
    >>>xs = np.r_[-np.flip(np.geomspace(1e-50, 1e30, 10000)), np.geomspace(1e-50, 1e30, 10000)]
    >>>scipy_results = scipy.special.wrightomega(xs)
    >>>torch_results = wrightomega(torch.tensor(xs))
    >>>torch.linalg.vector_norm(torch_results - scipy_results)
    tensor(1.7589e-13, dtype=torch.float64)

    """
    w = torch.zeros_like(x)
    w[x.isnan()] = float('nan')
    w[x.isinf() & (x > 0)] = float('inf')
    w[x.isinf() & (x < 0)] = 0
    finite_mask = torch.isfinite(x)

    tiny_mask = finite_mask & (x < -50)
    small_mask = finite_mask & (x >= -50) & (x < -2)
    w[small_mask | tiny_mask] = torch.exp(x[small_mask | tiny_mask])

    med_mask = finite_mask & (x >= -2) & (x < 1)
    w[med_mask] = torch.exp(2.0 * (x[med_mask] - 1.0) / 3.0);

    large_mask = finite_mask & (x >= 1) & (x < 1e20)
    lg = x[large_mask].log()
    w[large_mask] = x[large_mask] - lg + lg/x[large_mask];

    huge_mask = finite_mask & (x >= 1e20)
    w[huge_mask] = x[huge_mask]

    iterative_mask = small_mask | med_mask | large_mask

    # Iteration one of Fritsch, Shafer, and Crowley (FSC) iteration
    r = x[iterative_mask] - w[iterative_mask] - torch.log(w[iterative_mask]);
    wp1 = w[iterative_mask] + 1.0;
    e = (r / wp1) * (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - r) / (2.0 * wp1 * (wp1 + 2.0/3.0*r) - 2.0 * r);
    w[iterative_mask] = w[iterative_mask] * (1.0 + e);

    finfo = torch.finfo(w.dtype)
    wp1 = torch.zeros_like(x).masked_scatter(iterative_mask, wp1)
    r = torch.zeros_like(x).masked_scatter_(iterative_mask, r)
    next_iter_mask = torch.abs((2.0**w**w-8.0**w-1.0)*torch.pow(torch.abs(r),4.0)) >= finfo.tiny*72.0*torch.pow(torch.abs(wp1), 6.0)
    iterative_mask = iterative_mask & next_iter_mask

    # FSC iteration two
    r = x[iterative_mask] - w[iterative_mask] - torch.log(w[iterative_mask]);
    wp1 = w[iterative_mask] + 1.0;
    e = (r / wp1) * (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - r) / (2.0 * wp1 * (wp1 + 2.0/3.0*r) - 2.0 * r);
    w[iterative_mask] = w[iterative_mask] * (1.0 + e);

    return w