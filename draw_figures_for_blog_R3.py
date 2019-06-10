import numpy as np
from numpy import sqrt, pi
from conjugacy_math import *
from optimization import steepest_descent
from quadratic_form_contour_R3 import *
import plotly_tools as pt

A = np.array( [ [3.,1,2], [1,4,1], [2,1,7] ] ).reshape(3, 3)
b = np.array([4., 2, -3]).reshape(3, 1)
c = 0.
alpha = 1 / 2
beta = -1.
qfc = Quad_Form_Contour_R3()

x0 = np.array([-3., 4, 4]).reshape(3, 1)
evals, evecs = get_ordered_and_shifted_eigens_for_positive_definite_matrix(A)
center = compute_center(alpha, beta, b, evals, evecs).reshape(3, 1)
x_min = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)[1].reshape(3, 1)
assert np.allclose( center, x_min ), "center={} doesn't equal min point={} of f " \
    "as computed by scipy".format( center, x_min )

k = 1.
D_1, phi_1 = compute_Dk(1., A, b, c, alpha, beta)
M = evecs.dot(D_1).dot(evecs.T)
M_inv = np.linalg.inv(M)
assert np.allclose( M_inv, evecs.dot( np.linalg.inv(D_1) ).dot(evecs.T) ), 'M_inv is wrong'

def test1(debug_print=False):
    data = []
    ams = []
    k = 2. * pi

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, show_standard_basis=True, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)
    axes_max = np.max( np.abs(ams) )

    pt.plot_it_R3(data, axes_max, title='Fig.Test.1.R3', filename="Fig_Test_1_R3.html")
    return

def test2(num_iters=100, scale=100., debug_print=False):
    for i in range(num_iters):
        A = make_positive_definite_matrix(scale=scale, debug_print=debug_print)
        b = scale * (np.random.rand(3) * 2 - 1)
        c = np.random.uniform(low=-scale, high=scale)

        x0 = np.array([1., 1, 1])
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
    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        display_ellipsoid=False,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=debug_print
    )
    if debug_print: print("Test.AxesMax.1 axes_max={}".format(axes_max))
    assert np.abs(axes_max - 2.9193548387) < 1e-4, "incorrect axes_max={}. it should be 2.9193548387".format(axes_max)
    pt.plot_it_R3(data, axes_max, title='Test.AxesMax.1', filename="test_axes_max_01.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=debug_print
    )
    if debug_print: print("Test.AxesMax.2 axes_max={}".format(axes_max))
    assert np.abs(axes_max - 4.3164557013) < 1e-4, "incorrect axes_max={}. it should be 4.3164557013".format(axes_max)
    pt.plot_it_R3(data, axes_max, title='Test.AxesMax.2', filename="test_axes_max_02.html")
    return

def figure_ISD_2(debug_print=False):
    data = []
    ams = []

    steps = steepest_descent(A, b, x0) 
    if debug_print: print("Number of steps in Steepest Descent: {}".format( steps.shape[1] ))

    assert np.allclose( steps[:, -1], center.flatten() ), "steepest_descent didn't find the minimum point of f"
    for i in range( steps.shape[1] - 1 ):
        x_i = steps[:, i]
        r_i = b.flatten() - A.dot(x_i)
        x_ip1 = steps[:, i + 1]
        e_ip1 = x_ip1 - center.flatten()
        assert_A_orthogonal(A, r_i, e_ip1)

    trajectories = []
    for s in steps.T:
        r = b.flatten() - A.dot(s)
        rn = r / norm(r)
        if len(trajectories) == 0:
            trajectories.append( (rn, 1) )
            continue
        w = True
        for j, (t, cnt) in enumerate(trajectories):
            if np.allclose( rn, t, rtol=0., atol=1e-7 ):
                cnt += 1
                del trajectories[j]
                trajectories.insert( 0, (t, cnt) )
                w = False
                break
        if w: trajectories.append( (rn, 1) )

    if debug_print:
        print("Num steps = {} Num trajectories = {} Num dup = {}".format( steps.shape[1], len(trajectories), steps.shape[1] - len(trajectories) ))

    text = [py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(2)')]
    t2 = [''] * ( steps.shape[1] - 4 )
    text.extend(t2)
    text.append( py_text_sub('x', '(*)') )
    textposition = ['top right', 'top right', 'top right']
    tp2 = ['top center'] * len(t2)
    textposition.extend(tp2)
    textposition.append('top left')

    d = Scatter_R3(steps[0], steps[1], steps[2], mode='lines, text, markers', color='rgb(0,0,140)', width=3.,
            text=text, textposition=textposition, hoverinfo='x+y+z+text')
    data.append(d)

    r0 = b - A.dot(x0)
    alpha0 = innprd(r0, r0) / innprd(r0, A.dot(r0))
    x1 = x0 + alpha0 * r0
    assert np.allclose( x1.flatten(), steps[:, 1] ), "x1={} doesn't equal step[:,1]={}".format( x1.flatten(), steps[:, 1] )

    r0_normalized = r0.flatten() / norm(r0)
    d, _ = qfc.prep_vector_data( r0_normalized, center=x0, M=None, color='rgb(0,140,0)', name=py_text_sub('r', '(0)') )
    data.extend(d)

    e1 = x1 - center
    d, _ = qfc.prep_vector_data( e1, center=center, M=None, color='rgb(0,140,0)', name=py_text_sub('e', '(1)'), textposition='top right' )
    data.extend(d)

    d1, am1, E1 = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=debug_print)
    data.extend(d1)
    ams.append(am1)
    axes_max = np.max( np.abs(ams) )

    pt.plot_it_R3(data, axes_max, title='ISD.Fig.2', filename="Fig_ISD_2.html")
    return

def figures_PLE():
    wr = 1.5
    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='I',
        centered_at_origin=True,
        show_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, wr, title='PLE.Fig.1', filename="fig_PLE_01.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        display_ellipsoid=False,
        RTDR_variation='I',
        centered_at_origin=True,
        show_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, wr, title='PLE.Fig.2', filename="fig_PLE_02.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='R',
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, wr, title='PLE.Fig.3', filename="fig_PLE_03.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        RTDR_variation='DR',
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, wr, title='PLE.Fig.4', filename="fig_PLE_04.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        centered_at_origin=True,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, wr, title='PLE.Fig.5', filename="fig_PLE_05.html")

    data, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k,
        show_standard_basis=True,
        rotate_eigenvectors=True,
        rotate_standard_basis=True,
        debug_print=True
    )
    pt.plot_it_R3(data, axes_max, title='PLE.Fig.6', filename="fig_PLE_06.html")

    figure_PLE_7(debug_print=True)
    return

def figure_PLE_7(debug_print=False):
    data = []
    ams = []

    d, axes_max, E = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=debug_print)
    data.extend(d)

    phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
    sqrtkq = sqrt(phi_k / phi_1)
    name = 'f(x)={}'.format( rnd(k, 3) )
    dkq, amkq, Ekq = qfc.ellipsoid_from_ellipsoid_and_map( E, name=name, M=M_inv, center=center )
    data.append(dkq)
    for j in range(Ekq.shape[1]):
        w = norm( Ekq[:, j] - center.flatten() )
        assert np.abs( w - sqrtkq ) < 1e-9, "norm of Ekq = {} for col={} is wrong. S/b = {}".format(w, j, sqrtkq)

    pt.plot_it_R3(data, axes_max, title='PLE.Fig.7', filename="Fig_PLE_7.html")
    return

def figure_CJDR_2(debug_print=False):
    data = []
    ams = []
    k = 1.

    r0 = b - A.dot(x0)
    alpha0 = innprd(r0, r0) / innprd(r0, A.dot(r0))
    x1 = x0 + alpha0 * r0
    steps = np.hstack(( x0, x1, center ))

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(*)') ]
    textposition = ['top right', 'top right', 'top right']

    d = Scatter_R3(steps[0], steps[1], steps[2], mode='text, markers', color='rgb(0,0,140)', width=3.,
            text=text, textposition=textposition, hoverinfo='x+y+z+text')
    data.append(d)

    r0_normalized = r0.flatten() / norm(r0)
    d, _ = qfc.prep_vector_data( r0_normalized, center=x0, M=None, color='rgb(0,140,0)', name=py_text_sub('r', '(0)') )
    data.extend(d)

    Ar0 = A.dot(r0_normalized)
    d, _ = qfc.prep_vector_data( Ar0, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('Ar', '(0)') )
    data.extend(d)

    e1 = x1 - center
    d, _ = qfc.prep_vector_data( e1, center=center, M=None, color='rgb(0,140,0)', name=py_text_sub('e', '(1)'), textposition='top right' )
    data.extend(d)

    e1_normalized = e1.flatten() / norm(e1)
    assert_orthogonal(Ar0, e1_normalized)
    d, _ = qfc.prep_vector_data( e1_normalized, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('e', '(1)') )
    data.extend(d)

    assert_A_orthogonal(A, r0, -e1)
    d, _ = qfc.prep_vector_data( -e1_normalized, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('-e', '(1)') )
    data.extend(d)

    for i in range(1, 8):
        if i == 4: continue
        R = make_general_rotation_matrix_R3( angle=(i/4)*np.pi, axis = A.dot(r0) )
        oi = R.dot(e1)
        oi_normalized = oi / norm(oi)
        Ar0n = Ar0 / norm(Ar0)
        assert_orthogonal(Ar0n, oi_normalized, atol=1e-1)
        d, _ = qfc.prep_vector_data( oi_normalized, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('o', '({})'.format(i)) )
        data.extend(d)

    d, am, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=debug_print)
    data.extend(d)
    ams.append(am)
    axes_max = np.max( np.abs(ams) )

    pt.plot_it_R3(data, axes_max, title='CJDR.Fig.2', filename="Fig_CJDR_2.html")
    return

def get_plotly_data_for_level_k_and_its_dilated_ball(debug_print=False):
    data = []
    ams = []

    _, am, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, display_ellipsoid=True, debug_print=debug_print)
    ams.append(am)
    d, _, E = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, display_ellipsoid=False, debug_print=debug_print)
    data.extend(d)

    phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
    sqrtkq = sqrt(phi_k / phi_1)
    name = 'f(x)={}'.format( rnd(k, 3) )
    dkq, amkq, Ekq = qfc.ellipsoid_from_ellipsoid_and_map( E, name=name, M=M_inv, center=center )
    data.append(dkq)
    ams.append(amkq)
    for j in range(Ekq.shape[1]):
        w = norm( Ekq[:, j] - center.flatten() )
        assert np.abs( w - sqrtkq ) < 1e-9, "norm of Ekq = {} for col={} is wrong. S/b = {}".format(w, j, sqrtkq)

    axes_max = np.max( np.abs(ams) )
    return data, axes_max

def figure_CJDR_6(debug_print=False):
    data = []

    M_inv_x0 = M_inv.dot( x0 - center ) + center
    steps = np.hstack(( M_inv_x0, center ))

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(*)') ]
    textposition = ['top right', 'top left']

    d = Scatter_R3(steps[0], steps[1], steps[2], mode='text, markers', color='rgb(0,0,140)', width=3.,
            text=text, textposition=textposition, hoverinfo='x+y+z+text')
    data.append(d)

    d, axes_max = get_plotly_data_for_level_k_and_its_dilated_ball(debug_print=debug_print)
    data.extend(d)

    pt.plot_it_R3(data, axes_max, title='CJDR.Fig.6', filename="Fig_CJDR_6.html")
    return

def orthogonal_directions_with_standard_basis():
    M_inv_x0 = M_inv.dot( x0 - center ) + center
    M_inv_xw = center - M_inv_x0
    M_inv_x1 = np.array([ M_inv_xw[0, 0], 0, 0 ]).reshape(3, 1) + M_inv_x0
    M_inv_x2 = np.array([ M_inv_xw[0, 0], M_inv_xw[1, 0], 0 ]).reshape(3, 1) + M_inv_x0
    return np.hstack(( M_inv_x0, M_inv_x1, M_inv_x2, center ))

def figure_CJDR_7(debug_print=False):
    data = []
    steps = orthogonal_directions_with_standard_basis()

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(2)'), py_text_sub('x', '(*)') ]
    textposition = ['top right', 'top right', 'top right', 'top left']

    d = Scatter_R3(steps[0], steps[1], steps[2], mode='lines, text, markers', color='rgb(0,0,140)', width=3.,
            text=text, textposition=textposition, hoverinfo='x+y+z+text')
    data.append(d)

    d, axes_max = get_plotly_data_for_level_k_and_its_dilated_ball(debug_print=debug_print)
    data.extend(d)

    pt.plot_it_R3(data, axes_max, title='CJDR.Fig.7', filename="Fig_CJDR_7.html")
    return

def figure_CJDR_8(debug_print=False):
    data = []
    steps = orthogonal_directions_with_standard_basis()
    for i in range(3): 
        steps[:, i] = ( M.dot( steps[:, i].reshape(3, 1) - center ) + center ).reshape(3,)

    for i in range(3):
        u = steps[:, i + 1] - steps[:, i]
        for j in range(i + 1, 3):
            v = steps[:, j + 1] - steps[:, j]
            assert_A_orthogonal(A, u, v)

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(2)'), py_text_sub('x', '(*)') ]
    textposition = ['top right', 'top right', 'top right', 'top left']

    d = Scatter_R3(steps[0], steps[1], steps[2], mode='lines, text, markers', color='rgb(0,0,140)', width=3.,
            text=text, textposition=textposition, hoverinfo='x+y+z+text')
    data.append(d)

    d, axes_max, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k, debug_print=debug_print)
    data.extend(d)

    pt.plot_it_R3(data, axes_max, title='CJDR.Fig.8', filename="Fig_CJDR_8.html")
    return

def plot_all_R3_figures(debug_print=False):
    figure_ISD_2(debug_print=True)
    figures_PLE()
#    figure_PLE_7()
    figure_CJDR_2(debug_print=False)
    figure_CJDR_6(debug_print=False)
    figure_CJDR_7(debug_print=False)
    figure_CJDR_8(debug_print=False)
    return

def main():
#    test1(debug_print=True)
#    test2(debug_print=True)
#    test_axes_max(debug_print=True)
    plot_all_R3_figures(debug_print=False)
    return

if __name__ == "__main__":
    main()
