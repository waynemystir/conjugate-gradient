import numpy as np
from conjugacy_math import *
import optimization as optim
from plotly_tools import py_text_sub

class Quad_Form_Contour(object):

    def __init__(self,
            draw_vector_func,
            default_D,
            default_R,
            default_M,
            origin,
            default_x0,
            default_num_pts=40,
            default_RTDR_variation='RTDR',
        ):
        self._draw_vector_func = draw_vector_func
        self._default_D = default_D
        self._default_R = default_R
        self._default_M = default_M
        self._origin = origin
        self._default_x0 = default_x0
        self._default_num_pts = default_num_pts
        self._default_RTDR_variation = default_RTDR_variation
        return

    def ellipsoid_from_ellipsoid_and_map( self, E, name=None, color=None, M=np.eye(3), center=np.zeros((3,)), num_pts=40 ):
        raise NotImplementedError
    
    def ellipsoid_from_unit_ball_and_map( self, name=None, color=None, M=np.eye(3), center=np.zeros((3,)), num_pts=40 ):
        raise NotImplementedError
    
    def map_from_RTDR( self, D=None, R=None, RTDR_variation=None ):
        if RTDR_variation is None: RTDR_variation = self._default_RTDR_variation
        assert RTDR_variation in [ 'I', 'R', 'RTR', 'DR', 'RTDR' ], \
            "RTDR_variation={} not in 'I', 'R', 'RTR', 'DR', 'RTDR'".format(RTDR_variation)

        if D is None: D = self._default_D
        if R is None: R = self._default_R
        s = D.shape[0]

        if RTDR_variation == 'I': return np.eye(s)
        if RTDR_variation == 'R': return R
        if RTDR_variation == 'RTR': return np.eye(s)
        if RTDR_variation == 'DR': return D.dot(R)
        if RTDR_variation == 'RTDR': return R.T.dot(D).dot(R)
        return np.eye(3)
    
    def ellipsoid_from_unit_ball_and_RTDR_variation(
            self,
            name=None,
            color=None,
            D=None,
            R=None,
            RTDR_variation=None,
            center=None,
            num_pts=None
        ):
        if D is None: D = self._default_D
        if R is None: R = self._default_R
        if RTDR_variation is None: RTDR_variation = self._default_RTDR_variation
        center = self._origin.flatten() if center is None else center.flatten()
        if num_pts is None: num_pts = self._default_num_pts

        M = self.map_from_RTDR( D=D, R=R, RTDR_variation=RTDR_variation )
        return self.ellipsoid_from_unit_ball_and_map( name=name, color=color, M=M, center=center, num_pts=num_pts )
    
    def check_center_equals_min(self, center, x0, A, b, c, alpha, beta, atol=1e-05, check_CG=True, check_SD=True, debug_print=False):
        if debug_print:
            print("The strtpt is {}".format(x0.flatten()))
            print("The center is {}".format(center))
    
        _, x_min_sp = get_scipy_mins_for_quadratic_form(x0, A, b, c, alpha, beta)
        x_min_sp = np.array( x_min_sp ).reshape( x0.shape )
        if debug_print:
            print("The sp_min is {}".format(x_min_sp))
    
        cg_steps = optim.conjugate_gradient(A, b, x0)
        x_min_cg = cg_steps[:, -1]
        sd_steps = optim.steepest_descent(A, b, x0)
        x_min_sd = sd_steps[:, -1]
        if debug_print:
            print("The cg_min is {}".format(x_min_cg))
            print("The sd_min is {}".format(x_min_sd))
            print("Number of Conjugate Gradient steps = {}".format( cg_steps.shape[1] ))
            print("Number of Steepest Descent steps = {}".format( sd_steps.shape[1] ))
    
        assert np.allclose(center, x_min_sp, atol=atol), "center={} not equal to scipi min={}".format(center, x_min_sp)
        if check_CG: assert np.allclose(center, x_min_cg, atol=atol), "center not equal to min computed by conjugate gradient"
        if check_SD: assert np.allclose(center, x_min_sd, atol=atol), "center not equal to min computed by steepest descent"
        return
    
    def compute_rotation_to_standard_basis(self, vs, debug_print=False):
        """
        The goal of this function is to compute a matrix R that, when multiplied by the
        given vectors vs, gives us the standard basis vectors in R^n.
        """
        raise NotImplementedError
    
    def level_k_ellipsoid(self, A, b, c, alpha, beta, k,
            display_ellipsoid=True,
            ellipsoid_color=None,
            centered_at_origin=False,
            RTDR_variation='RTDR',
            show_eigenvectors=True,
            show_standard_basis=False,
            rotate_eigenvectors=False,
            rotate_standard_basis=False,
            eigenvector_color='rgb(0, 0, 255)',
            standard_basis_color='rgb(0, 140, 0)',
            debug_print=False,
            check_center=True,
            center_check_atol=1e-5
        ):
        evals, evecs = get_eigens_for_positive_definite_matrix(A)
        if debug_print: print("original evecs=\n{}".format(evecs))
        eigen_decomp = evecs.dot( np.diag(evals) ).dot(evecs.T)
        assert np.allclose(A, eigen_decomp), "A not equal to eigen_decomp"
    
        evals, evecs = order_and_shift_eigens(A, evals, evecs, debug_print=debug_print)
    
        phi_k = compute_phi_k(k, alpha, beta, b, c, evals, evecs)
        d = compute_semi_axes_lengths(phi_k, alpha, evals)
        D = np.diag(d)
        if debug_print: print("The semi-axes lengths are {}".format(d))
    
        computed_center = compute_center(alpha, beta, b, evals, evecs)
        if check_center:
            x0 = self._default_x0.reshape( computed_center.shape )
            self.check_center_equals_min(computed_center, x0, A, b, c, alpha, beta, atol=center_check_atol, debug_print=debug_print)
    
        center = self._origin if centered_at_origin else computed_center
    
        """
        Next we compute a map from the eigenvectors to the standard basis
        """
        R, es = self.compute_rotation_to_standard_basis(evecs, debug_print=debug_print)
    
        # R.T s/b first component in Eigendecomp
        eigen_decomp = R.T.dot( np.diag(evals) ).dot(R)
        assert np.allclose( A, eigen_decomp ), "A does not equal R^TSR\nA{}\ne={}".format(A, eigen_decomp)
        assert check_matrices_equal_except_column_sign(R.T, evecs), "R.T doesn't equal evecs\nR={}\ne={}".format(R, evecs)
        if debug_print: print( "R^T=\n{}\nevecs=\n{}".format(R.T, evecs) )
    
        """
        Now that we can easily map the eigenvectors to the standard basis vectors, we can just
        as easily go in reverse. We compute
        """
        R_inv = np.linalg.inv(R)
        assert np.allclose(R_inv, R.T), "R_inv doesn't equal R.T" # s/b true since R is orthogonal
        assert np.allclose(R_inv.dot(es), evecs), "R_inv times standard basis not equal to evecs"
        if debug_print: print("and R^T is a good map from the standard basis to the eigenvectors")
    
        '''
        Let's check that the points on our level k ellipsoid satisfy the quadratic form f(x)=k
        '''
        name = "f(x)={}".format( rnd(k, 1) ) if RTDR_variation == 'RTDR' and np.allclose(center, computed_center) else None
        ell, mxell, E = self.ellipsoid_from_unit_ball_and_RTDR_variation( name=name, D=D, R=R, color=ellipsoid_color, center=computed_center )
        for col in range( E.shape[1] ):
            x = E[:, col]
            f_x = quadratic_form(x, A, b, c, alpha, beta)
            assert np.abs(k - f_x) < 1e-8, "k={} and f_x={} are not equal".format(k, f_x)
        if debug_print: print("confirmed that f=k on level ellipsoid")
    
        '''
        Lastly, let's assemble the plotly data for the ellipse, vectors, and lines
        '''
        if display_ellipsoid and (RTDR_variation != 'RTDR' or centered_at_origin):
            ell, mxell, E = self.ellipsoid_from_unit_ball_and_RTDR_variation(
                name=name, D=D, R=R, color=ellipsoid_color, RTDR_variation=RTDR_variation, center=center )
    
        data = [] if not display_ellipsoid else [ell]
        axes_maxes = [center + 1., center - 1.] if not display_ellipsoid else [mxell]
        mr = np.max( axes_maxes )
    
        if debug_print: print("mr={} mxell={} all={}".format(mr, mxell, axes_maxes))
        if show_eigenvectors:
            M = self.map_from_RTDR( D=D, R=R, RTDR_variation=RTDR_variation ) if rotate_eigenvectors else None
            lds = self.prep_line_data_for_vectors(evecs, mr, center=center, M=M)
            data.extend(lds)
        elif show_standard_basis:
            M = self.map_from_RTDR( D=D, R=R, RTDR_variation=RTDR_variation ) if rotate_standard_basis else None
            lds = self.prep_line_data_for_vectors(es, mr, center=center, M=M)
            data.extend(lds)
    
        if show_eigenvectors:
            M = self.map_from_RTDR( D=D, R=R, RTDR_variation='RTR' if RTDR_variation=='RTDR' else 'R' ) if rotate_eigenvectors else None
            for j in range( evecs.shape[1] ):
                v, mx = self.prep_vector_data( evecs[:, j], center=center, M=M, color=eigenvector_color, name=py_text_sub('v', j + 1) )
                data.extend(v)
                axes_maxes.append(mx)
    
        if show_standard_basis:
            M = self.map_from_RTDR( D=D, R=R, RTDR_variation='RTR' if RTDR_variation=='RTDR' else 'R' ) if rotate_standard_basis else None
            for j in range( es.shape[1] ):
                v, mx = self.prep_vector_data( es[:, j], center=center, M=M, color=standard_basis_color, name=py_text_sub('e', j + 1) )
                data.extend(v)
                axes_maxes.append(mx)
    
        return data, mr, E
    
    def prep_vector_data(self, v, center=None, M=None, color='rgb(0, 0, 255)', name='', textposition=None):
        center = self._origin.flatten() if center is None else center.flatten()
        if M is None: M = self._default_M

        h = center + v.flatten()
        h = M.dot(h)
        return self._draw_vector_func(h, t=center, color=color, name=name, textposition=textposition)
    
    def prep_line_data_for_vectors(self, lvs, mr, center=np.zeros((3,)), M=np.eye(3), color='rgb(255, 0, 0)', width=3.):
        raise NotImplementedError
