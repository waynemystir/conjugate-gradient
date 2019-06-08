import numpy as np
from conjugacy_math import \
    make_positive_definite_matrix, \
    get_eigens_for_positive_definite_matrix, \
    get_scipy_mins_for_quadratic_form, \
    assert_orthogonal, \
    assert_A_orthogonal, \
    compute_Dk_for_x
from optimization import steepest_descent, conjugate_gradient

np.random.seed(41)

def test1():
    dim = 3
    alpha = 1/2 
    beta = -1. 
    scale = 100.

    A = make_positive_definite_matrix(scale=scale, dim=dim)
    evals, U = get_eigens_for_positive_definite_matrix(A)
    L = np.diag(evals)
    assert np.allclose(A, U.dot(L).dot(U.T)), "Eigendecomposition is wrong"
   
    b = scale * (np.random.rand(dim) * 2 - 1)
    c = np.random.uniform(low=-scale, high=scale)

    x0 = scale * (np.random.rand(dim) * 2 - 1)
    SD_steps = steepest_descent(A, b, x0)
    print("test1: number of steepest descent steps = {}".format( SD_steps.shape[1] ))

    x0c = SD_steps[:, 0]
    assert np.allclose(x0, x0c), "x0={} not equal to x0c={}".format(x0, x0c)

    xs = SD_steps[:, -1]
    _, xsc = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)
    assert np.allclose(xs, xsc), "xs={} not equal to xsc={}".format(xs, xsc)

    x1 = SD_steps[:, 1]
    x2 = SD_steps[:, 2]

    u = x1 - x0
    v = xs - x1
    w = x2 - x1
    x = xs - x2
    assert_A_orthogonal(A, u, v, atol=1e-5)
    assert_A_orthogonal(A, w, x, atol=1e-5)

    D_k, phi_k = compute_Dk_for_x(u, A, b, c, alpha, beta, evals, U)
    D_q, phi_q = compute_Dk_for_x(v, A, b, c, alpha, beta, evals, U)

    D_q_inv = np.linalg.inv(D_q)
    W = U.dot(D_q_inv).dot(U.T)
    print("D_q_inv={}".format(D_q_inv))
    print("W={}".format(W))
    a = W.dot(u)
    b = W.dot(v)
    assert_orthogonal(a, b)
    print("a={} and b={} are orthogonal".format(a, b))

    SD_steps_2 = steepest_descent(np.eye(dim), b, x)
    print("test1: number of steepest descent steps = {}".format( SD_steps_2.shape[1] ))
    ax0 = SD_steps_2[:, 0]
    ax1 = SD_steps_2[:, -1]
    print("x2={} xs={} ax0={} ax1={}".format(x2, xs, ax0, ax1))
    w0 = A.dot(ax0)
    w1 = A.dot(ax1)
    w1 = A.dot(ax1 + b)
    print("w0={} w1={}".format(w0, w1))
    return

def main():
    test1()
    return

if __name__ == '__main__':
    main()
