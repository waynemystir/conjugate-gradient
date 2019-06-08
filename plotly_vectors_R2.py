import numpy as np
from numpy import pi, cos, sin, arctan, arccos, sqrt
import plotly as py
import plotly.graph_objs as go

rnd = lambda x, n=2: np.round(x, n)
coords_equal = lambda x, y, tol=1e-11: np.abs(x - y) < tol
innprd=lambda u, v: u.flatten().dot( v.flatten() )
norm = lambda u: sqrt(innprd(u, u))

def comp_x_and_y1_and_y2(v1, v2, v3, debug_print=False):
    x = np.array( [v1[0], v2[0], v3[0]] )
    xa = np.argsort(x)
    x = np.sort(x)
    y = np.array( [v1[1], v2[1], v3[1]] )
    y = y[xa]
    y1s, y2s = np.zeros_like(x), np.zeros_like(x)

    if debug_print:
        print("x[0]={} x[1]={} x[2]={}".format(x[0], x[1], x[2]))
        print("y[0]={} y[1]={} y[2]={}".format(y[0], y[1], y[2]))

    for i in range(len(x)):
        y1_prev = y2_prev = y1 = y2 = 0
        xi = x[i]
        ymx = np.max(y)
        ymn = np.min(y)
        indices = np.argwhere( x==xi ).flatten()
        if i == 1 and y[0] < y[2] and ymx > y[2]:
            y2 = np.max(y)
            w1 = (x[2] - x[1]) / (x[2] - x[0])
            w2 = 1. - w1
            e1 = np.min(y)
            e2 = np.median(y)
            y1 = w1 * e1 + w2 * e2
            if debug_print:
                print("(I) i={} y[0]={} y1={} y[2]={} w1={}".format(i, y[0], y1, y[2], rnd(w1)))
        elif i == 1 and y[0] < y[2] and ymn < y[0]:
            y1 = np.min(y)
            w1 = (x[1] - x[0]) / (x[2] - x[0])
            w2 = 1. - w1
            e1 = np.max(y)
            e2 = np.median(y)
            y2 = w1 * e1 + w2 * e2
            if debug_print:
                print("(Ia) i={} w1={}".format(i, w1))
        elif i == 1 and y[0] > y[2] and ymx > y[0]:
            y2 = np.max(y)
            w1 = (x[1] - x[0]) / (x[2] - x[0])
            w2 = 1. - w1
            e1 = np.min(y)
            e2 = np.median(y)
            y1 = w1 * e1 + w2 * e2
            if debug_print:
                print("(Ib) i={} y[0]={} y[2]={} w1={}".format(i, y[0], y[2], rnd(w1)))
        elif i == 1 and y[0] > y[2] and ymn < y[2]:
            y1 = np.min(y)
            w1 = (x[2] - x[1]) / (x[2] - x[0])
            w2 = 1. - w1
            e1 = np.max(y)
            e2 = np.median(y)
            y2 = w1 * e1 + w2 * e2
            if debug_print:
                print("(Ic) i={} y[0]={} y[2]={} w1={}".format(i, y[0], y[2], rnd(w1)))
        elif i == 1 and coords_equal(x[0], x[1]):
            y1 = np.min(y)
            y2 = np.max(y)
#            if y[0] > y[1]:
#                y1 = y2 = np.min(y)
#            else:
#                y1 = y2 = np.max(y)
            if debug_print:
                print("(IIa) i={} y1={} y2={}".format(i, y1, y2))
        elif i == 1 and y[0] == y[2]:
            y1 = np.min(y)
            y2 = np.max(y)
            if debug_print:
                print("(IIb) i={} y1={} y2={}".format(i, y1, y2))
        elif len(indices) == 1:
            y1 = y2 = y[i]
            if debug_print:
                print( "(III) i={} LI=1 xi={} y1={} y2={}".format(i, xi, y1, y2) )
        else:
            y1 = y[ indices[0] ]
            y2 = y[ indices[1] ]
            if y1 > y2:
                tmp = y1
                y1 = y2
                y2 = tmp
            if debug_print:
                print( "(IV) i={} LI=2 xi={} y1={} y2={}".format(i, xi, y1, y2) )
        y1_prev = y1s[i] = y1
        y2_prev = y2s[i] = y2

    if debug_print:
        print( "xs={} y1sh={} y2sh={}".format(x.shape, y1s.shape, y2s.shape) )
        print("xa={}".format(xa))
    return x, y1s, y2s

def direction( h, t=np.zeros((2, 1)) ):
    v = h.flatten() - t.flatten()
    v = v / norm(v)
    a = 0
    if v[1] >= 0: a = arccos( v[0] )
    else: a = pi + arccos( -v[0] )

    if a < (0 * pi / 4 + 1 * pi / 4) / 2: return a, 'e'
    if (0 * pi / 4 + 1 * pi / 4) / 2 <= a and a < (1 * pi / 4 + 2 * pi / 4) / 2: return a, 'ne'
    if (1 * pi / 4 + 2 * pi / 4) / 2 <= a and a < (2 * pi / 4 + 3 * pi / 4) / 2: return a, 'n'
    if (2 * pi / 4 + 3 * pi / 4) / 2 <= a and a < (3 * pi / 4 + 4 * pi / 4) / 2: return a, 'nw'
    if (3 * pi / 4 + 4 * pi / 4) / 2 <= a and a < (4 * pi / 4 + 5 * pi / 4) / 2: return a, 'w'
    if (4 * pi / 4 + 5 * pi / 4) / 2 <= a and a < (5 * pi / 4 + 6 * pi / 4) / 2: return a, 'sw'
    if (5 * pi / 4 + 6 * pi / 4) / 2 <= a and a < (6 * pi / 4 + 7 * pi / 4) / 2: return a, 's'
    if (6 * pi / 4 + 7 * pi / 4) / 2 <= a and a < (7 * pi / 4 + 8 * pi / 4) / 2: return a, 'se'
    if (7 * pi / 4 + 8 * pi / 4) / 2 <= a and a < (8 * pi / 4 + 9 * pi / 4) / 2: return a, 'e'
    return a, 'ne'

def text_position_from_direction(d):
    if d == 'ne': return 'top right'
    if d == 'n': return 'top center'
    if d == 'nw': return 'top left'
    if d == 'w': return 'middle left'
    if d == 'sw': return 'bottom left'
    if d == 's': return 'bottom center'
    if d == 'se': return 'bottom right'
    if d == 'e': return 'middle right'
    return 'top right'

def draw_vector_R2(
        h,
        t=np.zeros((2, 1)),
        head_arrow_size=0.2,
        name=None,
        textposition=None,
        color='rgba(0, 0, 0, 1.0)',
        showlegend=False,
        hoverinfo=True,
        aspect_ratio=1.,
        debug_print=False
    ):
    t = t.flatten()
    h = h.flatten()
    v = h - t
    vc = np.array([v[0], (aspect_ratio  / 2) * v[1] ])
    vc = np.array([v[0], sqrt(aspect_ratio) * v[1] ])
    vc = v
    a, w = direction(vc)
    ca, sa = cos(a), sin(a)
    if debug_print:
        print("v={}\nvc={}".format(v, vc))
        print("aspect_ratio={} angle={} cos(a)={} sin(a)={}".format(aspect_ratio, a, ca, sa))
    d = head_arrow_size
    v2 = np.array([v[0] - 2*d, v[1] + .7*d])
    v3 = np.array([v[0] - 2*d, v[1] - .7*d])

    R = np.array([ca, -sa, sa, ca]).reshape(2, 2)
    v2 = R.dot(v2 - v) + v
    v3 = R.dot(v3 - v) + v

    x, y1, y2 = comp_x_and_y1_and_y2(v, v2, v3, debug_print=debug_print)
#    y1 = (2 / aspect_ratio) * (y1 - v[1]) + v[1]
#    y2 = (2 / aspect_ratio) * (y2 - v[1]) + v[1]
    if debug_print:
        print("x={}".format(x))
        print("y1={}".format(y1))
        print("y2={}".format(y2))

    x += t[0]
    y1 += t[1]
    y2 += t[1]

    tri1 = go.Scatter(x = x,
            y = y1,
            fill = None,
            line = dict(color = color),
            showlegend = False,
            hoverinfo = "none",
            mode= 'lines'
    )

    tri2 = go.Scatter(x = x,
            y = y2,
            fill = 'tonexty',
            fillcolor = color,
            line = dict(color = color),
            showlegend = False,
            hoverinfo = "none",
            mode = 'lines'
    )

    lx = np.array( [t[0], h[0]] )
    ly = np.array( [t[1], h[1]] )
    hi = "x+y" if hoverinfo else "none"
    tri3 = go.Scatter(
            x = lx,
            y = ly,
            line = dict(color = color),
            showlegend = showlegend,
            hoverinfo = hi,
            hoverlabel = dict(bgcolor = color),
            textfont = dict(color = color),
            name = name,
            text = ['', name],
            textposition=textposition or text_position_from_direction(w),
            mode = 'lines, text'
    )

    data = [tri1, tri2, tri3]
    mx = np.max( np.abs( [h, t] ) )
    return data, mx

def test1():
    x_start = 0.
    x_end = 11.17171
    y_start = 0.
    y_end = 3.3/2
    aspect_ratio = (x_end - x_start) / (y_end - y_start)

    v = np.array([6.,1])
#    data = draw_vector_R2(v, name="v", color='rgba(255,150,150,0.1)')[0]
#    data = draw_vector_R2(v, name="v", color='rgba(0,0,0,0.5)')[0]
    data = draw_vector_R2(v, t=np.array( [2., 1.4] ), name="v", aspect_ratio=aspect_ratio, debug_print=True)[0]
    layout = go.Layout(
        title = "hover on <i>points</i> or <i>fill</i>",
        xaxis = dict(
            range = [x_start, x_end]
        ),
        yaxis = dict(
#            scaleanchor = "x",
            range = [y_start, y_end]
        )
    )

    fig = go.Figure(data=data,layout=layout)
    py.offline.plot(fig, filename="plots/plot_vectors_1.html")
    return

def test2():
    data = []
    a = pi/4
    b, c = 3, 5
    v = np.array( [b * cos(a), c * sin(a)] )
    data.extend( draw_vector_R2(v, color='rgb(255,0,0)', head_arrow_size=0.6, name="red vector", debug_print=True, showlegend=True)[0] )

    a = 3*pi/4
    b, c = 4, 2
    v = np.array( [b * cos(a), c * sin(a)] )
    data.extend( draw_vector_R2(v, color='rgb(0,0,255)', head_arrow_size=0.6, name="blue vector", debug_print=True)[0] )

    m = np.max([b, c]) + 0.5
    x_start = -m
    x_end = m
    y_start = -m
    y_end = m

    layout = go.Layout(
        height=600,
        width=600,
        title = "<b>Test</b> <i>Number</i> <b>2</b>",
        xaxis = dict(
            range = [x_start, x_end]
        ),
        yaxis = dict(
            scaleanchor = "x",
            range = [y_start, y_end]
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename="plots/plot_vectors_2.html")
    return

def test3():
#    A = np.eye(2)
    A = np.array( [3.,2,2,6] ).reshape(2,2)
    w = np.linspace(0, 2 * pi, 60)
    data = []
    for i, a in enumerate(w):
        x = cos(a)
        y = sin(a)
        u = np.array([x, y])
        v = A.dot(5. * u)
        t = A.dot(u)
        data.extend( draw_vector_R2(v, t=t, head_arrow_size=0.39, name=r'w{}'.format(i), hoverinfo=False, debug_print=True)[0] )

    layout = go.Layout(
        height=800,
        width=800,
        title = "<b>Circle</b> of <i>Vectors</i>",
#        xaxis = dict(
#            range = [x_start, x_end]
#        ),
        yaxis = dict(
            scaleanchor = "x",
#            range = [y_start, y_end]
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename="plots/plot_vectors_3.html")
    return

def test4():
    m = 5
    r = [-m, m]
    layout = go.Layout(
        height=600,
        width=600,
        xaxis = dict(
            range = r
        ),
        yaxis = dict(
            scaleanchor = "x",
            range = r
        ),
        margin=dict(
            r=10, b=30,
            l=30, t=30)
    )

    def pdv(A, w=np.linspace(0, 2 * pi, 40, endpoint=False)):
        pdv.counter += 1
        data = []
        for i, a in enumerate(w):
            x = cos(a)
            y = sin(a)
            u = np.array([x, y])
            v = A.dot(2. * u)
            t = A.dot(u)
            c = 'rgb(255,0,0)' if i==0 \
                else 'rgb(0,255,0)' if i==10 \
                else 'rgb(0,0,255)' if i==20 \
                else 'rgb(0,255,255)' if i==30 \
                else 'rgb(0,0,0)'
            data.extend( draw_vector_R2(v, t=t, head_arrow_size=0.14, color=c, name=r'v{}'.format(i), hoverinfo=False, debug_print=True)[0] )
    
        fig = go.Figure(data=data, layout=layout)
        py.offline.plot(fig, filename="plots/plot_vectors_4_{}.html".format(pdv.counter))
        return
    pdv.counter = 0

    A = np.array( [1,1/2,1/2,2] ).reshape(2,2)
    U, S, V = np.linalg.svd(A)
#    U *= -1.
#    V *= -1.
    S = np.diag(S)
    pdv(np.eye(2))
#    pdv(A)
    pdv(V.T)
    pdv(S.dot(V.T))
    pdv(U.dot(S).dot(V.T))
    return

def test_all():
    test1()
    test2()
    test3()
    test4()
    return

def main():
#    test1()
#    test2()
#    test3()
#    test4()
    test_all()
    return

if __name__ == "__main__":
    main()
