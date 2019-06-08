from quadratic_form_contour import *
import numpy as np
from numpy import sin, cos, pi
from conjugacy_math import *
from plotly_vectors_R2 import draw_vector_R2
from plotly_tools import Scatter_R2

class Quad_Form_Contour_R2(Quad_Form_Contour):

    def __init__(self):
        super(Quad_Form_Contour_R2, self).__init__(
            draw_vector_R2,
            np.diag([4., 3.]),
            np.eye(2),
            np.eye(2),
            np.zeros((2,)),
            np.array( [-715.,349] ),
            default_num_pts=80
        )
        return

    def ellipsoid_from_ellipsoid_and_map( self, E, name=None, color=None, M=np.eye(2), center=np.zeros((2,)), num_pts=40 ):
        c = center.flatten()
        y = E[1]
    
        E = np.vstack(( E[0].flatten() - c[0], E[1].flatten() - c[1] ))
        E = M.dot(E)
        E = np.vstack(( E[0].flatten() + c[0], E[1].flatten() + c[1] ))
    
        hoverinfo = 'none' if name is None else 'name'
        if color is None: color = 'rgb(0,255,0)'
        ell = Scatter_R2(
            x = E[0].flatten(),
            y = E[1].flatten(),
            name = name,
            hoverinfo = hoverinfo,
            color=color,
            width=1.5,
            mode = 'lines, text'
        )
    
        axes_max = np.max( np.abs(E) )
        return ell, axes_max, E
    
    def ellipsoid_from_unit_ball_and_map( self, name=None, color=None, M=np.eye(2), center=np.zeros((2,)), num_pts=40 ):
        theta = np.linspace(0, 2 * pi, num=num_pts)
        x = cos(theta)
        y = sin(theta)
        E = np.vstack(( x.flatten(), y.flatten() ))
        c = center.flatten()
        E = np.vstack(( E[0].flatten() + c[0], E[1].flatten() + c[1] ))
        return self.ellipsoid_from_ellipsoid_and_map( E, name=name, color=color, M=M, center=c, num_pts=num_pts )
    
    def compute_rotation_to_standard_basis(self, vs, debug_print=False):
        """
        The goal of this function is to compute a matrix R that, when multiplied by the
        given vectors vs, gives us the standard basis vectors in R^2.
        """
        standard_basis = np.eye(2)
    
        a = angle_between( vs[:, 0], standard_basis[0] )
        a = -a if vs[1, 0] >= 0. else a
        if debug_print: print("angle_1={}".format(a))
        R = make_rotation_matrix_R2(angle=a, axis='x')
    
        es = R.dot(vs)
        assert check_matrices_equal_except_column_sign(es, standard_basis), \
            "es doesn't equal standard_basis_vectors\nes={}\nsb={}".format(es, standard_basis)
    
        if debug_print: print("completed rotation around x-axis")
        if debug_print: print("R is a good map from the eigenvectors to the standard basis")
        return R, es
    
    def prep_line_data_for_vectors(self, lvs, mr, center=np.zeros((2,)), M=np.eye(2), color='rgb(255, 0, 0)', width=1.2):
        c = self._origin.flatten() if center is None else center.flatten()
        if M is None: M = self._default_M

        data = []
        eps = [-mr-5, mr+5]
        lvs = M.dot(lvs)
    
        l = np.array( [c + ep * (lvs[:,0]) for ep in eps] )
        ls1 = Scatter_R2(l[:,0], l[:,1], color=color, width=width) 

        l = np.array( [c + ep * (lvs[:,1]) for ep in eps] )
        ls2 = Scatter_R2(l[:,0], l[:,1], color=color, width=width) 
    
        data.append(ls1)
        data.append(ls2)
        return data
