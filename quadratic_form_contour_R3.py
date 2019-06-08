from quadratic_form_contour import *
import numpy as np
from numpy import sqrt, sin, cos, pi, arccos
from conjugacy_math import *
from plotly.graph_objs import Mesh3d
from plotly_vectors_R3 import draw_vector_R3
from plotly_tools import Scatter_R3

class Quad_Form_Contour_R3(Quad_Form_Contour):

    def __init__(self):
        super(Quad_Form_Contour_R3, self).__init__(
            draw_vector_R3,
            np.diag([4., 3., 2.]),
            np.eye(3),
            np.eye(3),
            np.zeros((3,)),
            np.array( [-715.,349, -1] ),
        )
        return

    def ellipsoid_from_ellipsoid_and_map( self, E, name=None, color=None, M=np.eye(3), center=np.zeros((3,)), num_pts=40 ):
        c = center.flatten()
        z = E[2]
    
        E = np.vstack(( E[0].flatten() - c[0], E[1].flatten() - c[1], E[2].flatten() - c[2] ))
        E = M.dot(E)
        E = np.vstack(( E[0].flatten() + c[0], E[1].flatten() + c[1], E[2].flatten() + c[2] ))
    
        if color is None: color = [ [0.0, '#ff0000'], [0.5, '#00ff00'], [1.0, '#0000ff'] ]
        ell = Mesh3d(
            opacity = 0.4,
            colorscale = color,
            intensity = z.flatten(),
            name = 'z',
            hoverinfo = 'none',
            showscale = False,
            x = E[0].flatten(),
            y = E[1].flatten(),
            z = E[2].flatten(),
            alphahull = 0
        )
    
        axes_max = np.max( np.abs(E) )
        return ell, axes_max, E
    
    def ellipsoid_from_unit_ball_and_map( self, name=None, color=None, M=np.eye(3), center=np.zeros((3,)), num_pts=40 ):
        phi = np.linspace(0, 2 * pi, num=num_pts)
        theta = np.linspace(-pi / 2, pi / 2, num=num_pts)
        phi, theta = np.meshgrid(phi, theta)
        x = cos(theta) * sin(phi)
        y = cos(theta) * cos(phi)
        z = sin(theta)
        E = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        c = center.flatten()
        E = np.vstack(( E[0].flatten() + c[0], E[1].flatten() + c[1], E[2].flatten() + c[2] ))
        return self.ellipsoid_from_ellipsoid_and_map( E, name=name, color=color, M=M, center=c, num_pts=num_pts )
    
    def compute_rotation_to_standard_basis(self, vs, debug_print=False):
        """
        The goal of this function is to compute a matrix R that, when multiplied by the
        given vectors vs, gives us the standard basis vectors in R^3.
        """
        standard_basis = np.eye(3)
    
        v3_proj = orthogonal_projection(vs[:, 2], [standard_basis[0]])
        x00 = np.array([vs[0][2], 0., 0])
        assert np.allclose(v3_proj, x00), "x00 not equal to v3_projection"
    
        v3_dist_to_v3_proj = norm(vs[:, 2] - v3_proj)
        v3dd = vs[:, 2] / v3_dist_to_v3_proj
        a = pi / 2 - arccos( v3dd[1] )
        if debug_print: print("angle_1={}".format(a))
        R1 = make_rotation_matrix_R3(angle=a, axis='x')
    
        es = R1.dot(vs)
        if debug_print:
            print("the es")
            for j, e in enumerate(es):
                print("j={} e={}".format(j, e))
        assert np.abs(es[1, 2]) < 1e-9, "y-component of e3 isn't zero\nes=\n{}\nevecs=\n{}".format(es, vs)
        if debug_print: print("completed rotation around x-axis")
    
        if debug_print: print("compute angle between es2={} and k={}".format(es[:, 2], standard_basis[2]))
        a = angle_between(es[:, 2], standard_basis[2])
        a = a if es[0, 2] < 0. else -a
        if debug_print: print("angle_2={}".format(a))
        R2 = make_rotation_matrix_R3(angle=a, axis='y')
    
        es = R2.dot(R1).dot(vs)
        if debug_print:
            print("the es")
            for j, e in enumerate(es):
                print("j={} e={}".format(j, e))
        assert np.allclose( es[:, 2], standard_basis[2] ), \
            "es3={} is not equal to standard basis vec 3={}".format( es[:, 2], standard_basis[2] )
        if debug_print: print("completed rotation around y-axis")
    
        if debug_print: print("compute angle between es0={} and i={}".format(es[:, 0], standard_basis[0]))
        a = angle_between(es[:, 0], standard_basis[0])
        a = a if es[1, 0] <= 0. else -a
        if debug_print: print("angle_3={}".format(a))
        R3 = make_rotation_matrix_R3(angle=a, axis='z')
    
        R = R3.dot(R2).dot(R1)
        es = R.dot(vs)
        assert check_matrices_equal_except_column_sign(es, standard_basis), \
            "es doesn't equal standard_basis_vectors\nes={}\nsb={}".format(es, standard_basis)
    
        if debug_print: print("completed rotation around z-axis")
        if debug_print: print("R is a good map from the eigenvectors to the standard basis")
        return R, es
    
    def prep_line_data_for_vectors(self, lvs, mr, center=np.zeros((3,)), M=np.eye(3), color='rgb(255, 0, 0)', width=3.):
        c = self._origin.flatten() if center is None else center.flatten()
        if M is None: M = self._default_M

        data = []
        eps = [-mr-5, mr+5]
        lvs = M.dot(lvs)
    
        l = np.array( [c + ep * (lvs[:,0]) for ep in eps] )
        ls1 = Scatter_R3(l[:,0], l[:,1], l[:,2], color=color, width=width) 
        l = np.array( [c + ep * (lvs[:,1]) for ep in eps] )
        ls2 = Scatter_R3(l[:,0], l[:,1], l[:,2], color=color, width=width) 
        l = np.array( [c + ep * (lvs[:,2]) for ep in eps] )
        ls3 = Scatter_R3(l[:,0], l[:,1], l[:,2], color=color, width=width) 
    
        data.append(ls1)
        data.append(ls2)
        data.append(ls3)
        return data
