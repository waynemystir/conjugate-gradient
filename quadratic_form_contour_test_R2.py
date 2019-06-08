import numpy as np
from numpy import sqrt, pi
from conjugacy_math import *
from optimization import *
from quadratic_form_contour_R2 import *
import plotly_tools as pt

def test1(debug_print=False):
    A = np.array([3.,2,2,6]).reshape(2,2)
    b = np.array([2, -8])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []
    qfc = Quad_Form_Contour_R2()
    k = 2. * pi

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, show_standard_basis=True, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=2*k,
        ellipsoid_color='yellow', show_eigenvectors=False, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=3*k,
        ellipsoid_color='purple', show_eigenvectors=False, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=1.,
        ellipsoid_color='pink', show_eigenvectors=False, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)
    axes_max = np.max( np.abs(ams) )

    pt.plot_it_R2(data, axes_max, title='Fig.Test.1.R2', filename="Fig_Test_1_R2.html")
    return

def figure_ISD(debug_print=False):
    A = np.array([4.,3,3,7]).reshape(2,2)
    b = np.array([2, -13])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []
    qfc = Quad_Form_Contour_R2()
    ks = [18., 6., 1., -5., -9.]
    ecs = ['purple', 'green', 'pink', 'brown', 'orange']

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=12., ellipsoid_color='yellow', debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)

    for k, ec in zip(ks, ecs):
        d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k,
            ellipsoid_color=ec, show_eigenvectors=False, debug_print=debug_print)
        data.extend(d1)
        ams.append(am1)

    x0 = np.array([-3., -1])
    steps = steepest_descent(A, b, x0)
    cg_steps = conjugate_gradient(A, b, x0)
    if debug_print:
        print("figure_ISD: steepest-descent: start at x0={}".format(x0))
        print("number of steps for Steepest Descent={}\nnumber of steps for Conjugate Gradient={}".format(steps.shape[1], cg_steps.shape[1]))

    text = [pt.py_text_sub('x', '(0)'), pt.py_text_sub('x', '(1)')]
    t2 = [''] * ( steps.shape[1] - 3 )
    text.extend(t2)
    text.append( pt.py_text_sub('x', '(*)') )
    textposition = ['top center', 'top right']
    tp2 = ['top center'] * len(t2)
    textposition.extend(tp2)
    textposition.append('bottom center')

    l = pt.Scatter_R2(steps[0], steps[1], color='rgb(0,0,140)', width=1.,
                mode='lines, text, markers', text=text, textposition=textposition, textfontsize=18, hoverinfo='x+y')
    data.append(l)

    axes_max = np.max( np.abs(ams) )
    pt.plot_it_R2(data, axes_max, title='ISD.Fig.1', filename="Fig_ISD_1_R2.html", buffer_scale=1., buffer_fixed=.1)
    return

def figure_CJDR(debug_print=False):
    A = np.array([4.,3,3,7]).reshape(2,2)
    b = np.array([2, -13])
    c = 0
    alpha = 1 / 2
    beta = -1.

    D_1, phi_1 = compute_Dk(1., A, b, c, alpha, beta)
    evals, evecs = get_ordered_and_shifted_eigens_for_positive_definite_matrix(A, debug_print=debug_print)
    M = evecs.dot( np.linalg.inv(D_1) ).dot(evecs.T)
    center = compute_center(alpha, beta, b, evals, evecs)
    data = []
    ams = []

    qfc = Quad_Form_Contour_R2()
    ks = [18., 6., 1., -5., -9.]
    ecs = ['purple', 'green', 'pink', 'brown', 'orange']

    _, _, E = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=12., ellipsoid_color=None, debug_print=debug_print)
    d, am, E = qfc.ellipsoid_from_ellipsoid_and_map( E, name='wayne', color='yellow', M=M, center=center )
    data.append(d)
    ams.append(am)

    for k, ec in zip(ks, ecs):
        _, _, E = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k,
            ellipsoid_color=ec, show_eigenvectors=False, debug_print=debug_print)
        d, am, E = qfc.ellipsoid_from_ellipsoid_and_map( E, name='wayne', color=ec, M=M, center=center )
        data.append(d)
        ams.append(am)

    x0 = np.array([-3., -1])
    steps = steepest_descent(A, b, x0)
    cg_steps = conjugate_gradient(A, b, x0)
    if debug_print:
        print("figure_ISD: steepest-descent: start at x0={}".format(x0))
        print("number of steps for Steepest Descent={}\nnumber of steps for Conjugate Gradient={}".format(steps.shape[1], cg_steps.shape[1]))

    text = [pt.py_text_sub('x', '(0)'), pt.py_text_sub('x', '(1)')]
    t2 = [''] * ( steps.shape[1] - 3 )
    text.extend(t2)
    text.append( pt.py_text_sub('x', '(*)') )
    textposition = ['top center', 'top right']
    tp2 = ['top center'] * len(t2)
    textposition.extend(tp2)
    textposition.append('bottom center')

    l = pt.Scatter_R2(steps[0], steps[1], color='rgb(0,0,140)', width=1.,
                mode='lines, text, markers', text=text, textposition=textposition, textfontsize=18, hoverinfo='x+y')
    data.append(l)

    axes_max = np.max( np.abs(ams) )
    pt.plot_it_R2(data, axes_max, title='ISD.Fig.1', filename="Fig_ISD_1_R2.html", buffer_scale=1., buffer_fixed=.1)
    return

def test2(num_iters=100, scale=100., debug_print=False):
    alpha = 1 / 2
    beta = -1.
    qfc = Quad_Form_Contour_R2()

    for i in range(num_iters):
        A = make_positive_definite_matrix(dim=2, scale=scale, debug_print=debug_print)
        b = scale * (np.random.rand(2) * 2 - 1)
        c = np.random.uniform(low=-scale, high=scale)

        x0 = np.array([1., 1])
        fmin, _ = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)
        k = fmin + 1e-9 if i % 10 == 0 else np.random.uniform(low=fmin, high=fmin+400.)

        if debug_print:
            print("iter={}".format(i+1))
            print("A=\n{}".format(A))
            print("b={}".format(b))
            print("c={}".format(c))
            print("k={} fmin={}".format(k, fmin))

        _, _, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k, center_check_atol=.1, debug_print=debug_print)
    print("test2: completed {} iterations without a problem".format(i + 1))
    return

def test_axes_max(debug_print=False):
    A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3,3)
    b = np.array([4., 2, -3])
    c = 0
    k = 1.
    alpha = 1 / 2
    beta = -1.

    data, axes_max, _ = level_k_ellipsoid(A, b, c, alpha, beta, k,
        display_ellipsoid=False,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=debug_print
    )
    if debug_print: print("Test.AxesMax.1 axes_max={}".format(axes_max))
    assert np.abs(axes_max - 2.9193548387) < 1e-4, "incorrect axes_max={}. it should be 2.9193548387".format(axes_max)
    pt.plot_it_R2(data, axes_max, title='Test.AxesMax.1', filename="test_axes_max_01_R2.html")

    data, axes_max, _ = level_k_ellipsoid(A, b, c, alpha, beta, k,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=debug_print
    )
    if debug_print: print("Test.AxesMax.2 axes_max={}".format(axes_max))
    assert np.abs(axes_max - 4.3164557013) < 1e-4, "incorrect axes_max={}. it should be 4.3164557013".format(axes_max)
    pt.plot_it_R2(data, axes_max, title='Test.AxesMax.2', filename="test_axes_max_02_R2.html")
    return

def figures_PLE():
    A = np.array([3.,2,2,6]).reshape(2,2)
    b = np.array([2, -8])
    c = 0
    k = 1.
    alpha = 1 / 2
    beta = -1.
    qfc = Quad_Form_Contour_R2()

    wr = 1.5
    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='I',
        centered_at_origin=True,
        show_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, wr, title='PLE.Fig.1', filename="fig_PLE_01_R2.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        display_ellipsoid=False,
        RTDR_variation='I',
        centered_at_origin=True,
        show_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, wr, title='PLE.Fig.2', filename="fig_PLE_02_R2.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='R',
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, wr, title='PLE.Fig.3', filename="fig_PLE_03_R2.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='DR',
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, wr, title='PLE.Fig.4', filename="fig_PLE_04_R2.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, wr, title='PLE.Fig.5', filename="fig_PLE_05_R2.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R2(data, axes_max, title='PLE.Fig.6', filename="fig_PLE_06_R2.html")

    figure_PLE_Prop_4c(debug_print=True)
    return

def figure_PLE_Prop_4c(debug_print=False):
    A = np.array([3.,2,2,6]).reshape(2,2)
    b = np.array([2, -8])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []
    qfc = Quad_Form_Contour_R2()

    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    computed_center = compute_center(alpha, beta, b, evals, evecs)
    R = evecs.T
    L = np.diag( np.sqrt(evals) )

    axes_max = 4.9
    k = 2. * pi

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)

    q = 1. * pi
    phi_q = compute_phi_k(q, alpha, beta, b, c, evals, evecs)
    dq = compute_semi_axes_lengths(phi_q, alpha, evals)
    Dq = np.diag(dq)
    Dq_inv = np.linalg.inv(Dq)
    M = R.T.dot(Dq_inv).dot(R)

    phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
    sqrtkq = sqrt(phi_k / phi_q)
    name = 'radius={}'.format( rnd(sqrtkq, 1) )
    dkq, amkq, Ekq = qfc.ellipsoid_from_ellipsoid_and_map( E1, name=name, color='rgb(140,0,140)', M=M, center=computed_center )
    data.append(dkq)
    ams.append(amkq)
    for j in range(Ekq.shape[1]):
        w = norm( Ekq[:, j] - computed_center )
        assert np.abs( w - sqrtkq ) < 1e-9, "norm of Ekq = {} for col={} is wrong. S/b = {}".format(w, j, sqrtkq)

    pt.plot_it_R2(data, axes_max, title='Fig.PLE.Prop.4c_{}'.format(int(k)), filename="Fig_PLE_Prop_4c_{}_R2.html".format(int(k)))
    return

def test_DLE_2(debug_print=False):
    A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3,3)
    b = np.array([4., 2, -3])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []

    k = 1.
    d1, am1, E1 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)

    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    computed_center = compute_center(alpha, beta, b, evals, evecs)
    R = evecs.T
    Mev = R.T.dot(R)

    for i in [2., 7., 9.]:
        k = i
        phi_k = compute_phi_k(i, alpha, beta, b, c, evals, evecs)
        d = compute_semi_axes_lengths(phi_k, alpha, evals)
        D = np.diag(d)
    
        D_inv = np.linalg.inv(D)
        M = R.T.dot(D_inv).dot(R)
    
        dw, amw, Ew = ellipsoid_from_ellipsoid_and_map( E1, M=M, center=computed_center )
        data.append(dw)
        ams.append(amw)

    for j in range( evecs.shape[1] ):
        v, mx = prep_vector_data( evecs[:, j], center=computed_center, M=Mev, color='rgb(0,0,255)', name=r'v<sub>{}</sub>'.format(j + 1) )
        data.extend(v)
        ams.append(mx)

    axes_max = np.max(ams)
    lds = prep_line_data_for_vectors(evecs, axes_max, center=computed_center, M=M)
    data.extend(lds)
    pt.plot_it_R2(data, axes_max, title='Test.DLE.2', filename="Test_DLE_2_R2.html")
    return

def test_DLE_3(debug_print=False):
    A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3,3)
    A = np.array( [ [1.,1,2], [1,1.5,1], [2,1,.6] ] ).reshape(3,3)
    A = make_positive_definite_matrix( [2., 1.5, 3] )
    A = make_positive_definite_matrix( [6., 7.5, 8] )
    b = np.array([4., 2, -3])
    b = np.array([0., 2, -1])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []


    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    computed_center = compute_center(alpha, beta, b, evals, evecs)
    R = evecs.T
    L = np.eye(3)
    L = np.diag( np.sqrt(evals) )

    axes_max = 4.9
    for k in [1., 3., 9.]:
        d1, am1, E1 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)
        data.extend(d1)
        ams.append(am1)

        phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
        d = compute_semi_axes_lengths(phi_k, alpha, evals)
        D = np.diag(d)
    
        D_inv = np.linalg.inv(D)
        D_inv = np.eye(3)
        M = R.T.dot(L).dot(D_inv).dot(R)
    
        dw, amw, Ew = ellipsoid_from_ellipsoid_and_map( E1, M=M, center=computed_center )
        data.append(dw)
        ams.append(amw)

        pt.plot_it_R2(data, axes_max, title='Test.DLE.3_{}'.format(int(k)), filename="Test_DLE_3_{}_R2.html".format(int(k)))
        data, ams = [], []
    return

def test_DLE(debug_print=False):
    A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3,3)
    b = np.array([4., 2, -3])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []

    k = 1.
    d1, am1, E1 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)

    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    computed_center = compute_center(alpha, beta, b, evals, evecs)

    phi_1 = compute_phi_k(8, alpha, beta, b, c, evals, evecs)
    d = compute_semi_axes_lengths(phi_1, alpha, evals)
    D1 = np.diag(d)

    k = 2.
#    phi_2 = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
#    d = compute_semi_axes_lengths(phi_2, alpha, evals)
#    D2 = np.diag(d)

    R = evecs.T
    D = np.linalg.inv(D1)
    M = R.T.dot(D).dot(R)
    Mev = R.T.dot(R)

    d1, am1, E1 = ellipsoid_from_ellipsoid_and_map( E1, M=M, center=computed_center )
    data.append(d1)
    ams.append(am1)

    for j in range( evecs.shape[1] ):
        v, mx = prep_vector_data( evecs[:, j], center=computed_center, M=Mev, color='rgb(0,0,255)', name=r'v<sub>{}</sub>'.format(j + 1) )
        data.extend(v)
        ams.append(mx)

    axes_max = np.max(ams)
    lds = prep_line_data_for_vectors(evecs, axes_max, center=computed_center, M=M)
    data.extend(lds)

#    print("k={}".format(k))
#    d2, am2, E2 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)
#    d2, am2, E2 = ellipsoid_from_ellipsoid_and_map( E2, M=M, center=computed_center )
#    data.append(d2)
#    ams.append(am2)
#    axes_max = np.max(ams)

    pt.plot_it_R2(data, axes_max, title='DLE.Fig.1', filename="fig_DLE_01_R2.html")

    return

def figures_DLE(debug_print=False):
    A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3,3)
    b = np.array([4., 2, -3])
    c = 0
    alpha = 1 / 2
    beta = -1.
    data = []
    ams = []

    k = 1.
    d1, am1, E1 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)

    evals, evecs = get_eigens_for_positive_definite_matrix(A)
    evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    k = 24.
    phi_1 = compute_phi_k(1, alpha, beta, b, c, evals, evecs)
    d = compute_semi_axes_lengths(phi_1, alpha, evals)
    D1 = np.diag(d)
    phi_24 = compute_phi_k(k-7, alpha, beta, b, c, evals, evecs)
    d = compute_semi_axes_lengths(phi_24, alpha, evals)
    D24 = np.diag(d)
    D2s = np.diag([2.]*3)
    computed_center = compute_center(alpha, beta, b, evals, evecs)
    R = evecs.T
    D = np.linalg.inv(D1)
    D = D2s.dot( np.linalg.inv(D1) )
    M = R.T.dot(D).dot(R)
    Mev = R.T.dot(R)

    d1, am1, E1 = ellipsoid_from_ellipsoid_and_map( E1, M=M, center=computed_center )

    data.append(d1)
    ams.append(am1)

    for j in range( evecs.shape[1] ):
        v, mx = prep_vector_data( evecs[:, j], center=computed_center, M=Mev, color='rgb(0,0,255)', name=r'v<sub>{}</sub>'.format(j + 1) )
        data.extend(v)
        ams.append(mx)

    axes_max = np.max(ams)
    lds = prep_line_data_for_vectors(evecs, axes_max, center=computed_center, M=M)
    data.extend(lds)

    print("k={}".format(k))
    d2, am2, E2 = level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=True)
    D = np.linalg.inv(D24)
    M = R.T.dot(D).dot(R)
    d2, am2, E2 = ellipsoid_from_ellipsoid_and_map( E2, M=M, center=computed_center )
    data.append(d2)
    ams.append(am2)
    axes_max = np.max(ams)

    pt.plot_it_R2(data, axes_max, title='DLE.Fig.1', filename="fig_DLE_01_R2.html")

    return

def main():
#    test1(debug_print=True)
#    figure_ISD(debug_print=True)
#    figure_CJDR(debug_print=True)
#    test2(debug_print=True)
    figures_PLE()
#    figure_PLE_Prop_4c()
#    figures_DLE(debug_print=True)
#    test_DLE(debug_print=True)
#    test_DLE_2(debug_print=True)
#    test_DLE_3(debug_print=True)
#    test_axes_max(debug_print=True)
    return

if __name__ == "__main__":
    main()
