import numpy as np
from conjugacy_math import *

def check_check_are_orthogonal(u, v, check_assert=False):
    co = check_are_orthogonal(u, v)
    print("co[0]={} co[1]={}".format(co[0], co[1]))
    if check_assert: assert_orthogonal(u, v)
    return co[0]

def check_check_are_A_orthogonal(A, u, v, check_assert=False):
    co = check_are_A_orthogonal(A, u, v)
    print("co[0]={} co[1]={}".format(co[0], co[1]))
    if check_assert: assert_A_orthogonal(u, v)
    return co[0]

def test1():
    print("*************   test1   *************")

    u = np.array([1., 0])
    v = np.array([0., 1])
    assert check_check_are_orthogonal(u, v)
    u = np.array([1., 0])
    v = np.array([-1., 0])
    assert not check_check_are_orthogonal(u, v)
    m = np.array([-2, 2])
    assert not check_check_are_orthogonal(u + m, v + m)

    A = np.array([2., -1, -1, 2]).reshape(2, 2)
    assert not check_check_are_A_orthogonal(A, u, v)
    u = np.array([1., 0])
    v = np.array([0., 1])
    assert not check_check_are_A_orthogonal(A, u, v)
    print("A.dot( np.linalg.inv(A).dot(u) )={} and v={} should be orthogonal:".format( A.dot( np.linalg.inv(A).dot(u) ), v ) )
    assert check_check_are_A_orthogonal(A, np.linalg.inv(A).dot(u), v)
    assert check_check_are_A_orthogonal(A, u, np.linalg.inv(A).dot(v))

    input("Press Enter to continue...")
    return

def test2():
    print("*************   test2   *************")

    u = np.array([1., 0])
    v = np.array([0., 1])

    A = np.array([2., -1, -1, 2]).reshape(2, 2)
    evals, U = get_eigens_for_positive_definite_matrix(A)
    L = np.diag(evals)
    assert np.allclose(A, U.dot(L).dot(U.T)), "Eigendecomposition is wrong"

    alpha = 1/2
    beta = -1.
    b = np.array([3., -5])
    c = -7.

    Dk, _ = compute_Dk_for_x(u, A, b, c, alpha, beta, evals, U)
    Dq, _ = compute_Dk_for_x(v, A, b, c, alpha, beta, evals, U)
    UDkUTu = U.dot(Dk).dot(U.T).dot(u)
    UDqUTv = U.dot(Dq).dot(U.T).dot(v)
    assert check_check_are_A_orthogonal(A, UDkUTu, UDqUTv)

    u = np.random.rand(2)
    angle = angle_between(u, np.array([1., 0]))
    a = angle + np.pi / 2
    v = np.array([np.cos(a), np.sin(a)])
    assert check_check_are_orthogonal(u, v)

    Dk, _ = compute_Dk_for_x(u, A, b, c, alpha, beta, evals, U)
    Dq, _ = compute_Dk_for_x(v, A, b, c, alpha, beta, evals, U)
    UDkUTu = U.dot(Dk).dot(U.T).dot(u)
    UDqUTv = U.dot(Dq).dot(U.T).dot(v)
    assert check_check_are_A_orthogonal(A, UDkUTu, UDqUTv)

    input("Press Enter to continue...")
    return

def test3(num_iters=1000, debug_print=False):
    print("*************   test3   *************")
    alpha = 1/2
    beta = -1.
    scale = 100.

    for j in range(num_iters):
        u = np.array([1., 0, 0])
        v = np.array([0., 1, 0])
        R1 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='z')
        R2 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='x')
        R3 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='y')
        R = R1.dot(R2).dot(R3)
        u = R.dot(u)
        v = R.dot(v)
        assert_orthogonal(u, v)
        if debug_print: print('test3 iter=j u={} v={}'.format(j, u, v))
    
        A = make_positive_definite_matrix(scale=scale)
        evals, U = get_eigens_for_positive_definite_matrix(A)
        L = np.diag(evals)
        assert np.allclose(A, U.dot(L).dot(U.T)), "Eigendecomposition is wrong"
    
        b = scale * (np.random.rand(3) * 2 - 1)
        c = np.random.uniform(low=-scale, high=scale)
    
        Dk, phik = compute_Dk_for_x(u, A, b, c, alpha, beta, evals, U)
        Dq, phiq = compute_Dk_for_x(v, A, b, c, alpha, beta, evals, U)
        UDkUTu = U.dot(Dk).dot(U.T).dot(u)
        UDqUTv = U.dot(Dq).dot(U.T).dot(v)
        assert_A_orthogonal(A, UDkUTu, UDqUTv, atol=1e-7)
        w = innprd( A.dot(UDkUTu), UDqUTv )
        e = innprd(u, v) * sqrt(phik * phiq) / alpha
        assert np.abs(w - e) < 1e-7, "test3 iter={}, assert 4 failed: w={} e={}".format(j, w, e)
    print("completed {} iterations for test3 without a problem".format(j + 1))
    return

def test4(num_iters=1000, debug_print=False):
    print("*************   test4   *************")
    alpha = 1/2
    beta = -1.
    scale = 100.

    for j in range(num_iters):
        u = np.array([1., 0, 0])
        v = np.array([0., 1, 0])
        R1 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='z')
        R2 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='x')
        R3 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='y')
        R = R1.dot(R2).dot(R3)
        u = R.dot(u)
        R1 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='z')
        R2 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='x')
        R3 = make_rotation_matrix_R3(angle=np.random.uniform(0., 2 * pi), axis='y')
        R = R1.dot(R2).dot(R3)
        v = R.dot(v)
    
        A = make_positive_definite_matrix(scale=scale)
        evals, U = get_eigens_for_positive_definite_matrix(A)
        L = np.diag(evals)
        assert np.allclose(A, U.dot(L).dot(U.T)), "Eigendecomposition is wrong"
    
        b = scale * (np.random.rand(3) * 2 - 1)
        c = np.random.uniform(low=-scale, high=scale)
    
        Dk, phik = compute_Dk_for_x(u, A, b, c, alpha, beta, evals, U)
        Dq, phiq = compute_Dk_for_x(v, A, b, c, alpha, beta, evals, U)
        UDkUTu = U.dot(Dk).dot(U.T).dot(u)
        UDqUTv = U.dot(Dq).dot(U.T).dot(v)
        w = innprd( A.dot(UDkUTu), UDqUTv )
        e = innprd(u, v) * sqrt(phik * phiq) / alpha
        if debug_print: print('test4 iter={} u={} v={} angle={} w={} e={}'.format(j, rnd(u), rnd(v), rnd( angle_between(u, v) ), rnd(w), rnd(e)))
        assert np.abs(w - e) < 1e-6, "test4, assert 2 failed: w={} e={}".format(w, e)
    print("completed {} iterations for test4 without a problem".format(j + 1))
    return

def test_all(debug_print=False):
    test1()
    test2()
    test3(debug_print=debug_print)
    test4(debug_print=debug_print)
    return

def main():
    test_all(debug_print=True)
    return

if __name__ == "__main__":
    main()
