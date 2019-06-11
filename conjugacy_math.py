import numpy as np
from numpy import sqrt, cos, sin, arccos, pi
from scipy import optimize as sopt

innprd = lambda u, v: u.flatten().dot( v.flatten() )
norm = lambda v: sqrt( innprd(v, v) )
angle_between = lambda u, v: arccos( innprd(u, v) / (norm(u) * norm(v)) )
are_orthogonal = lambda u, v, atol=1e-9: np.abs( innprd(u, v) ) < atol
are_A_orthogonal = lambda A, u, v, atol=1e-9: are_orthogonal( A.dot(u), v, atol=atol )
orthogonal_projection = lambda v, es: np.sum( [innprd(v, e) * e for e in es], axis=0 )
is_symmetric = lambda A: np.allclose(A, A.T, atol=0.)
rnd = lambda x, n=4: np.round(x, n)
triangular_number = lambda n: int( (n + 1) * n / 2 )
factorial = lambda n: np.prod( [i for i in range(1, n + 1)] )

def check_are_orthogonal(u, v, atol=1e-9):
    if are_orthogonal(u, v, atol=atol): return True, "u={} and v={} are orthogonal".format(u, v)
    return False, "u={} and v={} are not orthogonal. angle is {}".format(u, v, angle_between(u, v))

def check_are_A_orthogonal(A, u, v, atol=1e-9):
    if are_A_orthogonal(A, u, v, atol=atol): return True, "u={} and v={} are A-orthogonal".format(u, v)
    return False, "u={} and v={} are not A-orthogonal. angle is {}".format(u, v, angle_between(A.dot(u), v))

def assert_orthogonal(u, v, atol=1e-9):
    co = check_are_orthogonal(u, v, atol=atol)
    assert co[0], co[1]
    return

def check_all_columns_orthogonal(U, atol=1e-9, print_non_orthogonal=False):
    results = []
    for j in range(U.shape[1] - 1):
        for i in range(1, U.shape[1] - j):
            co = check_are_orthogonal(U[:, j], U[:, j + i], atol=atol) 
            r = (j, j + i, co)
            results.append(r)

    tUn = triangular_number( U.shape[1] - 1 )
    assert len(results) == tUn, "message_all_columns_orthogonal: didn't collect correct " \
        "number of results. Should have {}. Got {}".format( tUn, len(results) )

    if print_non_orthogonal:
        for (c1, c2, mo) in results:
            if not mo[0]: print("Columns {} and {} are NOT orthogonal:\n{}".format( c1, c2, mo[1] ))
    return results

def assert_all_columns_orthogonal(U, atol=1e-9):
    aaco = True
    results = check_all_columns_orthogonal(U, atol=atol)
    for (c1, c2, mo) in results:
        if not mo[0]:
            aaco = False
            print("Columns {} and {} are NOT orthogonal:\n{}".format( c1, c2, mo[1] ))
    assert aaco, "Above columns are not orthogonal"
    return

def assert_A_orthogonal(A, u, v, atol=1e-9):
    co = check_are_A_orthogonal(A, u, v, atol=atol)
    assert co[0], co[1]
    return

def assert_is_eigenpair(A, l, v):
    assert np.allclose( A.dot(v), l * v ), "v={} is not an eigenvector of A corresponding to l={}".format(v, l)

def assert_are_eigenpairs(A, evals, evecs, debug_print=False, title=None):
    if title is not None: print(title)
    for j, (l, v) in enumerate(zip(evals, evecs.T)):
            assert_is_eigenpair(A, l, v)
            if debug_print: print("completed eigenpair check for j={} eigenvalue={} eigenvector={}".format(j, l, v))
    return

def get_eigens_for_positive_definite_matrix(A):
    assert is_symmetric(A), "A is not symmetric"
    evals, evecs = np.linalg.eig(A)
    assert len( np.where(evals <= 0.)[0] ) == 0, "A is not positive definite. The eigenvalues are {}".format(evals)
    return evals, evecs

def order_and_shift_eigens(A, evals, evecs, debug_print=False):
    assert_are_eigenpairs(A, evals, evecs, debug_print=debug_print, title="unordered eigenpairs" if debug_print else None)
    assert_all_columns_orthogonal(evecs)

    order = np.argsort(evals)
    evals = evals[ order ]
    evecs = evecs.T[ order ].T 
    assert_are_eigenpairs(A, evals, evecs, debug_print=debug_print, title="ordered eigenpairs" if debug_print else None)

    for j in range( evecs.shape[1] ):
        if evecs[j, j] < 0.: evecs[:, j] *= -1. 

    assert_are_eigenpairs(A, evals, evecs, debug_print=debug_print, title="shifted eigenpairs" if debug_print else None)
    assert_all_columns_orthogonal(evecs)

    return evals, evecs

def get_ordered_and_shifted_eigens_for_positive_definite_matrix(A, debug_print=False):
    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    return order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)

def make_positive_definite_matrix(diag=None, dim=3, scale=100., positive_definite_scale = .4, debug_print=False):
    rn = lambda: scale * (np.random.rand() * 2 - 1)
    if diag is None:
        A = np.zeros((dim, dim))
        for i in range(A.shape[0]):
            A[i, i] = np.abs(rn())
    else:
        A = np.diag(diag)
        scale = np.min(diag)

    w = 0
    while True:
        w += 1
        for i in range(A.shape[0]):
            for j in range(i):
                if i != j: A[i, j] = A[j, i] = positive_definite_scale * rn()
        assert is_symmetric(A), "make_positive_definite_matrix: made matrix A that is not symmetric: A=\n{}".format(A)
        evals, _ = np.linalg.eig(A)
        if len( np.where(evals <= 0.)[0] ) > 0: continue # skip if A isn't positive definite
        if debug_print: print("make_positive_definite_matrix: succeeded after {} attempts".format(w))
        break
    return A

def quadratic_form(x, A, b, c, alpha, beta):
    return alpha * innprd(A.dot(x), x) + beta * innprd(b, x) + c

def compute_phi_k(k, alpha, beta, b, c, evals, evecs):
    phik = k - c + np.sum( [ (beta * innprd(b, v))**2 / (4 * alpha * l) for l, v in zip(evals, evecs.T) ] ) 
    assert phik > 0., "For k={}, we computed phi_k={} <= 0. But phi_k must be positive. " \
        "Choose k > min f(x) over all x and this error won't occur.".format(k, phik)
    return phik

def compute_semi_axes_lengths(phi_k, alpha, evals):
    return [ sqrt( phi_k / (alpha * l) ) for l in evals ]

def compute_center(alpha, beta, b, evals, evecs):
    cent = lambda lam, evec: ( -1 * beta * innprd(b, evec) / (2 * alpha * lam) ) * evec
    return np.sum([ cent(lam, evec) for lam, evec in zip(evals, evecs.T) ], axis=0)

def compute_Dk(k, A, b, c, alpha, beta, evals=None, evecs=None):
    if evals is None or evecs is None:
        evals, evecs = get_ordered_and_shifted_eigens_for_positive_definite_matrix(A)
    phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
    dk = compute_semi_axes_lengths(phi_k, alpha, evals)
    return np.diag(dk), phi_k

def compute_Dk_for_x(x, A, b, c, alpha, beta, evals=None, evecs=None):
    k = quadratic_form(x, A, b, c, alpha, beta)
    return compute_Dk(k, A, b, c, alpha, beta, evals, evecs)

def compute_radius_after_dilating_ellipse_to_ball(alpha, beta, b, c, evals, evecs, k=None, q=None, phi_k=None, phi_q=None):
        if phi_k is None:
            assert k is not None, "This function requires a value for k or phi_k. Both are None."
            phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
        if phi_q is None:
            assert q is not None, "This function requires a value for q or phi_q. Both are None."
            phi_q = compute_phi_k(q, alpha, beta, b, c, evals, evecs)
        return sqrt( phi_k / phi_q )

def quadratic_gradient(x, A, b, c, alpha, beta, d=.4, num_pts=100):
    mpts = int(num_pts / 2)
    x1 = np.linspace( x[0] - d, x[0] + d, num_pts )
    x2 = np.linspace( x[1] - d, x[1] + d, num_pts )
    x3 = np.linspace( x[2] - d, x[2] + d, num_pts )
    xm = np.meshgrid(x1, x2, x3)
    fx = np.zeros((num_pts, num_pts, num_pts))
    for i in range(num_pts):
        for j in range(num_pts):
            for k in range(num_pts):
                v = np.array([ x1[i], x2[j], x3[k] ])
                fx[i, j, k] = quadratic_form(v, A, b, c, alpha, beta)
    dx1, dx2, dx3 = np.gradient(fx)
    print("dx1=\n{}\ndx2=\n{}\ndx3=\n{}".format( dx1, dx2, dx3 ))
    print("x={}\nx1={}\nx2={}\nx3={}\nfx=\n{}".format(x, x1, x2, x3, fx))
    dfx = np.array( [dx1[mpts, mpts, mpts], dx2[mpts, mpts, mpts], dx3[mpts, mpts, mpts]] )
    return dfx

def get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta):
    f_x = lambda x: quadratic_form(x.reshape( b.shape ), A, b, c, alpha, beta)
    minf = sopt.minimize( f_x, x0 )
    f_min = minf['fun']
    x_min = minf['x'].reshape( b.shape )
    return f_min, x_min

def make_rotation_matrix_R3(angle=pi/4, axis='x'):
    assert axis in ['x', 'y', 'z'], "make_rotation_matrix_R3: axis ({}) not 'x', 'y', or 'z'".format(axis)
    c, s = cos(angle), sin(angle)
    if axis=='x':
        R = np.array([1., 0, 0, 0, c, -s, 0, s, c]).reshape(3,3)
    elif axis=='y':
        R = np.array([c, 0., s, 0, 1, 0, -s, 0, c]).reshape(3,3)
    elif axis=='z':
        R = np.array([c, -s, 0., s, c, 0, 0, 0, 1]).reshape(3,3)
    return R

def make_general_rotation_matrix_R3( angle=pi/4, axis=np.array([1., 1, 1]) ):
    n = axis.flatten() / norm(axis)
    n1, n2, n3 = n
    c, s, omc = cos(angle), sin(angle), 1. - cos(angle)
    r1 = np.array([ c + n1**2 * omc, n1 * n2 * omc - n3 * s, n1 * n3 * omc - n2 * s ])
    r2 = np.array([ n1 * n2 * omc + n3 * s, c + n2**2 * omc, n2 * n3 * omc - n1 * s ])
    r3 = np.array([ n1 * n3 * omc - n2 * s, n2 * n3 * omc + n1 * s, c + n3**2 * omc ])
    return np.vstack(( r1, r2, r3 ))

def make_rotation_matrix_R2(angle=pi/4, axis='x'):
    assert axis in ['x', 'y'], "make_rotation_matrix_R2: axis ({}) not 'x' or 'y'".format(axis)
    c, s = cos(angle), sin(angle)
    if axis=='x':
        R = np.array([c, -s, s, c]).reshape(2,2)
    elif axis=='y':
        R = np.array([c, -s, s, c]).reshape(2,2)
    return R

def vector_at_angle_R2(v, angle):
    v = v / norm(v)
    R = make_rotation_matrix_R2(angle=angle, axis='x')
    u = R.dot(v).flatten()
    return u

def check_matrices_equal_except_column_sign(A, B): 
    A, B = np.array(A), np.array(B)
    if np.allclose(A, B): return True
    for i in range(len(A)):
        a = A[:, i]
        b = B[:, i]
        if not np.allclose(a, b) and not np.allclose(a, -b):
            return False
    return True

def gram_schmidt_conjugation(A, us):
    if type(us) == list:
        us = [u.reshape(-1, 1) for u in us]
        us = np.hstack(us)

    # TODO: check for linear independence of us
    ds = []
    for u in us.T:
        s = np.sum( [ d * innprd(u, A.dot(d)) / innprd(d, A.dot(d))  for d in ds ], axis=0 )
        ds.append(u - s)
    return ds

def test_check_all_columns_orthogonal():
    print("************** test_check_all_columns_orthogonal: starting **************\n")

    A = make_positive_definite_matrix(dim=5)
    evals, evecs = np.linalg.eig(A)
    print("evals={}".format(evals))
    print("evecs=\n{}".format(evecs))
    assert_all_columns_orthogonal(evecs )
    print("************** test_check_all_columns_orthogonal: first test passed **************\n")

    A = np.eye(5)
    A[:, 1] = np.array([ 1., 2,3,4,5] )
    print("A=\n{}".format(A))
    check_all_columns_orthogonal(A, print_non_orthogonal=True)
    print("************** test_check_all_columns_orthogonal: second test done: should have shown some columns not orthogonal**************\n")

    A = np.arange(1,17).reshape(4,4)
    print("A=\n{}".format(A))
    check_all_columns_orthogonal(A, print_non_orthogonal=True)
    print("************** test_check_all_columns_orthogonal: third test done: should have shown all columns not orthogonal**************\n")

    evals, evecs = np.linalg.eig(A)
    print("evals={}".format(evals))

    check_all_columns_orthogonal(evecs, print_non_orthogonal=True)
    print("************** test_check_all_columns_orthogonal: fourth test done: should have shown some columns not orthogonal**************\n")

    print("************** test_check_all_columns_orthogonal: fifth test should throw assertion error: **************\n")
    assert_all_columns_orthogonal(evecs)
    return

def test_rotation_matrix_R2_a(v=None):
    standard_basis = np.eye(2)
    sb1 = standard_basis[0]
    if v is None: v = np.array([1.,1])
    print("v={}".format(v))
    v = v / norm(v)
    a = angle_between( v, sb1 )
    print("angle={}".format(a))
    a = -a if v[1] >= 0. else a
    R = make_rotation_matrix_R2(angle=a, axis='x')
    print("R={}".format(R))
    u = R.dot(v)
    print("u={}".format(u))
    assert np.allclose( u, sb1 ), "rotation u={} to expected standard basis vector={} failed".format(u, sb1)
    return

def test_rotation_matrix_R2():
    v = np.array([-1.,-1])
    v = np.array([-1.,1])
    test_rotation_matrix_R2_a(v=v)
    return

def main():
#    test_check_all_columns_orthogonal()
    test_rotation_matrix_R2()
    return

if __name__ == "__main__":
    main()
