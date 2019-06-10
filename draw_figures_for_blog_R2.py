from numpy import \
    array as np_array, \
    sqrt as np_sqrt, \
    max as np_max, \
    abs as np_abs, \
    hstack as np_hstack, \
    allclose as np_allclose
from numpy.linalg import inv
from numpy.random import uniform as np_uniform, rand as np_rand
from quadratic_form_contour_R2 import Quad_Form_Contour_R2
from optimization import steepest_descent
from plotly_tools import py_text_sub, Scatter_R2, plot_it_R2
from conjugacy_math import \
    compute_Dk, \
    compute_phi_k, \
    get_ordered_and_shifted_eigens_for_positive_definite_matrix, \
    compute_center, \
    norm, \
    rnd, \
    compute_radius_after_dilating_ellipse_to_ball, \
    assert_A_orthogonal, \
    get_scipy_mins_for_quadratic_form, \
    innprd, \
    angle_between, \
    make_positive_definite_matrix, \
    gram_schmidt_conjugation

A = np_array([4., 3, 3, 7]).reshape(2, 2)
b = np_array([2, -13]).reshape(2, 1)
c = 0.
alpha = 1 / 2 
beta = -1. 
x0 = np_array([-3.5, -0.5]).reshape(2, 1)
evals, evecs = get_ordered_and_shifted_eigens_for_positive_definite_matrix(A)
center = compute_center(alpha, beta, b, evals, evecs).reshape(2, 1)
_, x_min = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)
assert np_allclose( center, x_min ), "center={} doesn't equal min point={} of f " \
    "as computed by scipy".format( center.flatten(), x_min )

qfc = Quad_Form_Contour_R2()
ks = [18., 12., 6., 1., -5., -9.]
ecs = ['purple', 'yellow', 'green', 'pink', 'brown', 'orange']

D_1, phi_1 = compute_Dk(1., A, b, c, alpha, beta)
M = evecs.dot(D_1).dot(evecs.T)
M_inv = inv(M)
assert np_allclose( M_inv, evecs.dot( inv(D_1) ).dot(evecs.T) ), 'M_inv is wrong'

def test1(num_iters=100, scale=100., debug_print=False):
    for i in range(num_iters):
        A = make_positive_definite_matrix(dim=2, scale=scale, debug_print=debug_print)
        b = scale * (np_rand(2) * 2 - 1)
        c = np_uniform(low=-scale, high=scale)

        x0 = np_array([1., 1]) 
        fmin, _ = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)
        k = fmin + 1e-9 if i % 10 == 0 else np_uniform(low=fmin, high=fmin+400.)

        if debug_print:
            print("iter={}".format(i+1))
            print("A=\n{}".format(A))
            print("b={}".format(b))
            print("c={}".format(c))
            print("k={} fmin={}".format(k, fmin))

        _, _, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k, center_check_atol=.1, debug_print=debug_print)
    print("test1: completed {} iterations without a problem".format(i + 1)) 
    return

def draw_concentric_contours(steps, text, textposition, title, filename, plotly_data=[], steps_mode='lines, text, markers', debug_print=False):
    ply_data = []
    if plotly_data is not None: ply_data.extend( plotly_data )
    ams = []

    if steps is None and debug_print:
        print("No steps")
    elif debug_print:
        print("{}: steepest-descent: start at x0={}".format(title, x0))
        print("number of steps for Steepest Descent={}".format(steps.shape[1]))

    sef = True
    for k, ec in zip(ks, ecs):
        d, am, _ = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k,
            ellipsoid_color=ec, show_eigenvectors=sef, debug_print=debug_print)
        sef = False
        ply_data.extend(d)
        ams.append(am)

    if steps is not None:
        steps_ply_data = Scatter_R2( steps[0], steps[1], color='rgb(0,0,140)', width=1.,
                    mode=steps_mode, text=text, textposition=textposition, textfontsize=18, hoverinfo='x+y' )
        ply_data.append(steps_ply_data)

    axes_max = np_max( np_abs(ams) )
    plot_it_R2(ply_data, axes_max, title=title, filename=filename, buffer_scale=1., buffer_fixed=.1)
    return

def draw_concentric_balls(steps, text, textposition, title, filename, steps_mode='lines, text, markers', debug_print=False):
    ply_data = []
    ams = []

    if debug_print:
        print("{}: steepest-descent: start at x0={}".format(title, x0))
        print("number of steps for Steepest Descent={}".format(steps.shape[1]))

    for k, ec in zip(ks, ecs):
        expected_radius = compute_radius_after_dilating_ellipse_to_ball(alpha, beta, b, c, evals, evecs, k=k, phi_q=phi_1)

        _, am, E = qfc.level_k_ellipsoid(A, b, c, alpha, beta, k=k,
            ellipsoid_color=ec, show_eigenvectors=False, debug_print=debug_print)
        ams.append(am)
        d, am, E = qfc.ellipsoid_from_ellipsoid_and_map( E, name='radius={}'.format( rnd(expected_radius, 1) ), color=ec, M=M_inv, center=center )

        for x in E.T:
            computed_radius = norm( x - center.flatten() )
            assert np_abs( computed_radius - expected_radius ) < 1e-9, \
                "computed radius={} doesn't equal expected radius={}".format( computed_radius, expected_radius )

        ply_data.append(d)
        ams.append(am)

    steps_ply_data = Scatter_R2( steps[0], steps[1], color='rgb(0,0,140)', width=1.,
                mode=steps_mode, text=text, textposition=textposition, textfontsize=18, hoverinfo='x+y' )
    ply_data.append(steps_ply_data)

    axes_max = np_max( np_abs(ams) )

    lds = qfc.prep_line_data_for_vectors(evecs, axes_max, center=center)
    ply_data.extend(lds)
    for i in range(2):
        d, _ = qfc.prep_vector_data( evecs[:, i], center=center, M=None, name=py_text_sub('v', i + 1) )
        ply_data.extend(d)

    plot_it_R2(ply_data, axes_max, title=title, filename=filename, buffer_scale=1., buffer_fixed=.1)
    return

def figure_ISD_1(debug_print=False):
    steps = steepest_descent(A, b, x0) 
    assert np_allclose( steps[:, -1], center.flatten() ), "steepest_descent didn't find the minimum point of f"
    for i in range( steps.shape[1] - 1 ):
        x_i = steps[:, i]
        r_i = b.flatten() - A.dot(x_i)
        x_ip1 = steps[:, i + 1]
        e_ip1 = x_ip1 - center.flatten()
        assert_A_orthogonal(A, r_i, e_ip1)

    text = [py_text_sub('x', '(0)'), py_text_sub('x', '(1)')]
    t2 = [''] * ( steps.shape[1] - 3 ) 
    text.extend(t2)
    text.append( py_text_sub('x', '(*)') )
    textposition = ['middle left', 'top right']
    tp2 = ['top center'] * len(t2)
    textposition.extend(tp2)
    textposition.append('bottom center')

    plotly_data = []

    r0 = b - A.dot(x0)
    alpha0 = innprd(r0, r0) / innprd(r0, A.dot(r0))
    x1 = x0 + alpha0 * r0

    r0_normalized = r0.flatten() / norm(r0)
    d, _ = qfc.prep_vector_data( r0_normalized, center=x0, M=None, color='rgb(0,140,0)', name=py_text_sub('r', '(0)') )
    plotly_data.extend(d)

    e1 = x1 - center
    d, _ = qfc.prep_vector_data( e1, center=center, M=None, color='rgb(0,140,0)', name=py_text_sub('e', '(1)'), textposition='bottom left' )
    plotly_data.extend(d)

    draw_concentric_contours(steps, text, textposition, plotly_data=plotly_data, title='ISD.Fig.1', filename='Fig_ISD_1.html', debug_print=debug_print)
    return

def figure_CJDR_1(debug_print=False):
    r0 = b - A.dot(x0)
    alpha0 = innprd(r0, r0) / innprd(r0, A.dot(r0))
    x1 = x0 + alpha0 * r0
    steps = np_hstack(( x0, x1, center ))

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(*)') ]
    textposition = [ 'middle left', 'middle left', 'bottom center' ]

    plotly_data = []

    r0_normalized = r0.flatten() / norm(r0)
    d, _ = qfc.prep_vector_data( r0_normalized, center=x0, M=None, color='rgb(0,140,0)', name=py_text_sub('r', '(0)') )
    plotly_data.extend(d)

    Ar0 = A.dot(r0_normalized)
    d, _ = qfc.prep_vector_data( Ar0, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('Ar', '(0)') )
    plotly_data.extend(d)

    e1 = x1 - center
    d, _ = qfc.prep_vector_data( e1, center=center, M=None, color='rgb(0,140,0)', name=py_text_sub('e', '(1)'), textposition='top right' )
    plotly_data.extend(d)

    e1_normalized = e1.flatten() / norm(e1)
    d, _ = qfc.prep_vector_data( e1_normalized, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('e', '(1)') )
    plotly_data.extend(d)

    assert_A_orthogonal(A, r0, -e1)
    d, _ = qfc.prep_vector_data( -e1_normalized, center=x0, M=None, color='rgb(170,0,0)', name=py_text_sub('-e', '(1)') )
    plotly_data.extend(d)

    draw_concentric_contours(steps, text, textposition, plotly_data=plotly_data,
        title='CJDR.Fig.1', filename='Fig_CJDR_1.html', steps_mode='text, markers', debug_print=debug_print)
    return

def figure_CJDR_3(debug_print=False):
    M_inv_x0 = M_inv.dot( x0 - center ) + center
    steps = np_hstack(( M_inv_x0, center ))

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(*)') ]
    textposition = [ 'middle left', 'bottom center' ]

    draw_concentric_balls(steps, text, textposition, title='CJDR.Fig.3', filename='Fig_CJDR_3.html', steps_mode='text, markers', debug_print=debug_print)
    return

def orthogonal_directions_with_standard_basis():
    M_inv_x0 = M_inv.dot( x0 - center ) + center
    M_inv_xw = center - M_inv_x0
    M_inv_x1 = np_array([ M_inv_xw[0, 0], 0 ]).reshape(2, 1) + M_inv_x0
    return np_hstack(( M_inv_x0, M_inv_x1, center ))

def figure_CJDR_4(debug_print=False):
    steps = orthogonal_directions_with_standard_basis()

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(*)') ]
    textposition = [ 'middle left', 'top center', 'bottom center' ]

    draw_concentric_balls(steps, text, textposition, title='CJDR.Fig.4', filename='Fig_CJDR_4.html', debug_print=debug_print)
    return

def figure_CJDR_5(debug_print=False):
    steps = orthogonal_directions_with_standard_basis()
    for i in [0, 1]:
        steps[:, i] = ( M.dot( steps[:, i].reshape(2, 1) - center ) + center ).reshape(2,)

    u = steps[:, 1] - steps[:, 0]
    v = steps[:, 2] - steps[:, 1]
    assert_A_orthogonal(A, u, v)

    text = [ py_text_sub('x', '(0)'), py_text_sub('x', '(1)'), py_text_sub('x', '(*)') ]
    textposition = [ 'middle left', 'top center', 'bottom center' ]

    draw_concentric_contours(steps, text, textposition, title='CJDR.Fig.5', filename='Fig_CJDR_5.html', debug_print=debug_print)
    return

def figure_GSC_1(debug_print=False):
    u = np_array([1., 0]).reshape(2, 1)
    v = np_array([0., 5]).reshape(2, 1)
    us = np_hstack((u, v))
    d0, d1 = gram_schmidt_conjugation(A, us)
    print("d0={} d1={}".format(d0, d1))
    assert_A_orthogonal(A, d0, d1)
    print("yep")
    return

def plot_all_R2_figures(debug_print=False):
    figure_ISD_1(debug_print=debug_print)
    figure_CJDR_1(debug_print=debug_print)
    figure_CJDR_3(debug_print=debug_print)
    figure_CJDR_4(debug_print=debug_print)
    figure_CJDR_5(debug_print=debug_print)
    figure_GSC_1(debug_print=debug_print)
    return

def main():
#    test1(debug_print=True)
    plot_all_R2_figures(debug_print=False)
    return

if __name__ == '__main__':
    main()
