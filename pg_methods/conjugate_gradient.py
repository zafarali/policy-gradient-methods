"""Implementation of the conjugate gradient algorithm."""
import numpy as np

def conjugate_gradient_algorithm(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312
    https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/trpo.py#L122
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):

        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x