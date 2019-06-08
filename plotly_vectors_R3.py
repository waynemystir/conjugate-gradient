import numpy as np
import plotly
import plotly.graph_objs as go

def draw_vector_R3(h, t=np.zeros((3, )), color='rgb(0, 0, 0)', name='', arrow_size=0., showlegend=False, textposition=None):

    q = np.linalg.norm(h - t)
    f = lambda q: np.exp( np.power(q / 2.5, 2.5/9) )

    if arrow_size == 0.:
        a = .99 - np.exp(-f(q))
        arrow_size = .6 * (a**(1.7))

    g = lambda q: 1. - np.exp(-0.85 * q)
    q = .99 - np.exp(-g(q) * f(q))

    x, y, z = h
    u, v, w = h - t
    cone = go.Cone(
        x = [x],
        y = [y],
        z = [z],
        u = [u],
        v = [v],
        w = [w],
        anchor = "tip",
        sizemode = "absolute",
        sizeref = arrow_size,
        name = name,
        hoverinfo = 'x+y+z+name',
        hoverlabel = dict(bgcolor = color),
        colorscale = [[0, color], [0.5, color], [1.0, color]],
        showscale = False
    )

    lx = [t[0], t[0] + q * (h[0] - t[0])]
    ly = [t[1], t[1] + q * (h[1] - t[1])]
    lz = [t[2], t[2] + q * (h[2] - t[2])]
    l = go.Scatter3d(x = lx, y = ly, z = lz,
        showlegend = showlegend,
        text = ['', name],
        textfont = dict(color = color),
        hoverinfo = 'none',
        line=dict(
            color=color,
#            color='rgb(0, 255, 0)',
            width=9,
        ),
        mode = 'lines, text'
    )

    mx = np.max( np.abs( [h, t] ) )

    data = [l, cone]
    return data, mx

def get_3d_vectors_test_layout(r, title=''):
    layout = go.Layout(
        title = go.layout.Title(
            text = title,
            xref = 'paper',
            x = 0,
            y = .99 
        ),
        autosize = False,
        width = 600,
        height = 600,
        margin = dict(
            l = 0,
            r = 0,
            b = 0,
            t = 0,
            pad = 4),
        scene = dict(
            aspectratio = go.layout.scene.Aspectratio(x=1, y=1, z=1),
            xaxis = dict(
                range=r,
                title='x'),
            yaxis = dict(
                range=r,
                title='y'),
            zaxis = dict(
                range=r,
                title='z' ))
    )
    return layout

def test1(arrow_size=.6):
#    head = np.array([1., 1, 1])
#    head = np.array([1., 3, 1])
    head = np.array([1., 1, 2])
#    head = np.array([5., 77, -712])
#    tail = np.array([0., 0, 0])
#    tail = np.array([19., 1, 1])
    tail = np.array([-17., 3, 1])
    data, mx1 = draw_vector_R3(head, tail, color='rgb(255, 0, 0)', name=r'v<sub>1</sub>', arrow_size=arrow_size)
#    layout = {
#        'scene': {
#          'camera': {
#            'eye': {'x': -0.76, 'y': 1.8, 'z': 0.92}
#          }
#        }
#    }

    head = np.array([1., 3, 1])
    tail = np.array([-27., 7, 13])
    d2, mx2 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>2</sub>', arrow_size=arrow_size)
    data.extend(d2)

    head = np.array([1., 0, 2])
    d3, mx3 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>3</sub>', arrow_size=arrow_size, showlegend=True)
    data.extend(d3)

    head = np.array([1., -1, 2])
    tail = np.array([-0.2, -1, 2])
    d4, mx4 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>4</sub>', arrow_size=arrow_size, showlegend=True)
    data.extend(d4)

    head = np.array([44., -1, 2])
    tail = np.array([42., -1, 2])
    d5, mx5 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>5</sub>', arrow_size=arrow_size)
    data.extend(d5)

    head = np.array([1., -2, 2])
    tail = np.array([-1.2, -2, 2])
    d6, mx6 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>6</sub>', arrow_size=arrow_size)
    data.extend(d6)

    head = np.array([1., -3, 2])
    tail = np.array([-2.2, -3, 2])
    d7, mx7 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>7</sub>', arrow_size=arrow_size)
    data.extend(d7)

    head = np.array([1., -4, 2])
    tail = np.array([-5.2, -4, 2])
    d8, mx8 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>8</sub>', arrow_size=arrow_size)
    data.extend(d8)

    head = np.array([1., -5, 2])
    tail = np.array([-33.2, -5, 2])
    d9, mx9 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>9</sub>', arrow_size=arrow_size)
    data.extend(d9)

    head = np.array([1., 5, 2])
    tail = np.array([0.05, 5, 2])
    d10, mx10 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>10</sub>', arrow_size=arrow_size)
    data.extend(d10)

    head = np.array([1., 6, 2])
    tail = np.array([0.25, 6, 2])
    d11, mx11 = draw_vector_R3(head, tail, color='rgb(255, 0, 0)', name=r'v<sub>11</sub>', arrow_size=arrow_size, showlegend=True)
    data.extend(d11)

    head = np.array([1., 7, 2])
    tail = np.array([0.5, 7, 2])
    d12, mx12 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>12</sub>', arrow_size=arrow_size, showlegend=True)
    data.extend(d12)

    head = np.array([1., 8, 2])
    tail = np.array([0.7, 8, 2])
    d13, mx13 = draw_vector_R3(head, tail, color='rgb(255, 0, 0)', name=r'v<sub>13</sub>', arrow_size=arrow_size, showlegend=True)
    data.extend(d13)

    mx = np.max([mx1, mx2, mx3, mx4, mx5]) + 2.
    r = [-mx, mx]
    fig = go.Figure(data=data, layout=get_3d_vectors_test_layout(r, title='Test-1'))
    plotly.offline.plot(fig, filename='plots/plotly-vectors-3d-test-1.html', validate=False)
    return

def test2(arrow_size=.3):
    head = np.array([1., 1, 1])
    data, mx1 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>1</sub>', arrow_size=arrow_size)

    head = np.array([.75, -.5, 0])
    tail = np.array([-.5, -.5, 0])
    d2, mx2 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>2</sub>', arrow_size=arrow_size)
    data.extend(d2)

    head = np.array([-.25, .25, 0])
    d3, mx3 = draw_vector_R3(head, color='rgb(200, 200, 0)', name=r'v<sub>3</sub>', arrow_size=arrow_size)
    data.extend(d3)

    head = np.array([-.5, -.5, .7])
    tail = np.array([-.5, -.5, 0])
    d4, mx4 = draw_vector_R3(head, tail, color='rgb(0, 0, 255)', name=r'v<sub>4</sub>', arrow_size=arrow_size)
    data.extend(d4)

    head = np.array([0., 0, 1])
    d5, mx5 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>5</sub>', arrow_size=arrow_size)
    data.extend(d5)

    head = np.array([-1., -1, -1])
    head /= np.linalg.norm(head)
    d6, mx6 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>5</sub>', arrow_size=arrow_size)
    data.extend(d6)

    mx = np.max( [mx1, mx2, mx3, mx4] ) + 1.
    r = [-mx, mx]
    fig = go.Figure(data=data, layout=get_3d_vectors_test_layout(r, title='Test-2'))
    plotly.offline.plot(fig, filename='plots/plotly-vectors-3d-test-2.html', validate=False)
    return

def test3(arrow_size=.076):
    head = np.array([.4, .4, .4])
    data, mx1 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>1</sub>', arrow_size=arrow_size)

    head = np.array([-.1, .1, .1])
    d2, mx2 = draw_vector_R3(head, color='rgb(0, 0, 255)', name=r'v<sub>2</sub>', arrow_size=arrow_size)
    data.extend(d2)

    mx = np.max( [mx1, mx2] ) + .1
    r = [-mx, mx]
    fig = go.Figure(data=data, layout=get_3d_vectors_test_layout(r, title='Test-3'))
    plotly.offline.plot(fig, filename='plots/plotly-vectors-3d-test-3.html', validate=False)
    return

def test4(arrow_size=.076):
    head = np.array([-0.62796303, -0.45970084, -0.62796303])
    data, mx1 = draw_vector_R3(head, color='rgb(255, 0, 0)', name=r'v<sub>1</sub>', arrow_size=arrow_size)

    mx = 6.
    r = [-mx, mx]
    fig = go.Figure(data=data, layout=get_3d_vectors_test_layout(r, title='Test-4'))
    plotly.offline.plot(fig, filename='plots/plotly-vectors-3d-test-4.html', validate=False)
    return

def test5():
    data, ms = [], []
    xs = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1., 1.01, 1.05, 1.1, 1.3, 1.5, 2., 3., 4., 5., 6., 7., 9., 11., 13., 15., 17., 20.]
    xs.reverse()
    ys = [-6 + .5 * j for j in range( len(xs) )]
    for i, (x, y) in enumerate(zip(xs, ys)):
        head = np.array([1., y, 2])
        tail = np.array([1. - x, y, 2])
        w = i % 2
        c = 'rgb(0, 0, 255)' if w == 0 else 'rgb(0, 140, 0)'
        d, m = draw_vector_R3(head, tail, color=c, name=r'v<sub>{}</sub>'.format(i+1))
        data.extend(d)
        ms.append(m)

    mx = np.max(ms) + 2.
    r = [-mx, mx]
    fig = go.Figure(data=data, layout=get_3d_vectors_test_layout(r, title='Test-5'))
    plotly.offline.plot(fig, filename='plots/plotly-vectors-3d-test-5.html', validate=False)
    return

def test_all(arrow_size=0.):
#    test1(arrow_size=arrow_size)
    test2(arrow_size=arrow_size)
    test3(arrow_size=arrow_size)
    test4(arrow_size=arrow_size)
    test5()
    return

def main():
#    test1()
#    test1(arrow_size=0.)
#    test2()
#    test2(arrow_size=0.)
#    test3()
#    test3(arrow_size=0.)
#    test4(arrow_size=0.)
    test_all()
    return

if __name__ == "__main__":
    main()
