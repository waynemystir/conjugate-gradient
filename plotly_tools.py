import numpy as np
import plotly
import plotly.graph_objs as go
from plotly import tools

def plot_it_R3(data, axes_max=3., title='', filename='level_ellipsoid_R3_1.html', buffer_scale=1.1, buffer_fixed=1.):
    mr = buffer_scale * axes_max + buffer_fixed
    v1 = vertical_axis_3D(mr)
    data.append(v1)
    r = [-mr, mr]

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

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="./plots/{}".format(filename), include_plotlyjs='cdn')
    return

def plot_it_R2(data, axes_max=3., title='', filename='level_ellipsoid_R2_1.html', buffer_scale=1.1, buffer_fixed=1.):
    mr = buffer_scale * axes_max + buffer_fixed
#    v1 = vertical_axis_3D(mr)
#    data.append(v1)
    r = [-mr, mr]

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
        hovermode = 'closest',
        margin = dict(
            l = 40,
            r = 40,
            b = 40,
            t = 40,
            pad = 4
        ),
#        aspectratio = go.layout.scene.Aspectratio(x=1, y=1),
        xaxis = dict(
            range=r,
            title='x',
            showticklabels=True,
#            dtick=1.,
#            tickmode='linear',
            nticks=18,
        ),
        yaxis = dict(
            scaleanchor = "x",
            range=r,
            title='y',
            showticklabels=True,
#            dtick=1.,
#            tickmode='linear',
            nticks=18,
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="./plots/{}".format(filename), include_plotlyjs='cdn')
    return

def plot_it_R2_short_3in1(data1, data2, data3, axes_max1=3., axes_max2=3., axes_max3=3., nticks=10,
        title='', filename='short_3in1_R2.html', buffer_scale=1.1, buffer_fixed=1.):
    mr1 = buffer_scale * axes_max1 + buffer_fixed
    r1 = [-mr1, mr1]
    mr2 = buffer_scale * axes_max2 + buffer_fixed
    r2 = [-mr2, mr2]
    mr3 = buffer_scale * axes_max3 + buffer_fixed
    r3 = [-mr3, mr3]

    fig = tools.make_subplots(rows=1, cols=3)
    for d in data1: fig.append_trace(d, 1, 1)
    for d in data2: fig.append_trace(d, 1, 2)
    for d in data3: fig.append_trace(d, 1, 3)

    fig['layout']['title'].update(text=title)
    fig['layout']['title'].update(xref='paper')
    fig['layout']['title'].update(x=0)
    fig['layout']['title'].update(y=.99)
    fig['layout']['margin'].update(l=40)
    fig['layout']['margin'].update(r=40)
    fig['layout']['margin'].update(b=40)
    fig['layout']['margin'].update(t=40)
    fig['layout']['margin'].update(pad=4)
    fig['layout'].update(autosize=False)
    fig['layout'].update(height=390)
    fig['layout'].update(width=900)
    fig['layout'].update(hovermode='closest')
    fig['layout']['xaxis1'].update(range=r1)
    fig['layout']['xaxis1'].update(nticks=nticks)
    fig['layout']['yaxis1'].update(range=r1)
    fig['layout']['yaxis1'].update(nticks=nticks)
    fig['layout']['yaxis1'].update(scaleanchor='x1')
    fig['layout']['xaxis2'].update(range=r2)
    fig['layout']['xaxis2'].update(nticks=nticks)
    fig['layout']['yaxis2'].update(range=r2)
    fig['layout']['yaxis2'].update(nticks=nticks)
    fig['layout']['yaxis2'].update(scaleanchor='x2')
    fig['layout']['xaxis3'].update(range=r3)
    fig['layout']['xaxis3'].update(nticks=nticks)
    fig['layout']['yaxis3'].update(range=r3)
    fig['layout']['yaxis3'].update(nticks=nticks)
    fig['layout']['yaxis3'].update(scaleanchor='x3')

    plotly.offline.plot(fig, filename="./plots/{}".format(filename), include_plotlyjs='cdn')
    return

def vertical_axis_3D(axes_max, marker_size=2.):
    v1_z = np.linspace(-axes_max, axes_max, 100)
    v1_x = np.zeros_like(v1_z)
    v1_y = v1_x.copy()
    v1 = go.Scatter3d(x = v1_x, y = v1_y, z = v1_z,
        showlegend = False,
        name = 'z-axis',
        hoverinfo = 'x+y+z+name',
        marker=dict(
            size=marker_size,
            color=v1_z,
            colorscale='Viridis',
        ),  
        line=dict(
            color='#1f77b4',
            width=1
        )   
    )   
    return v1

def Scatter_R3(x, y, z,
        mode='lines',
        name=None,
        color='rgb(255,0,0)',
        width=3.,
        text=None,
        textcolor=None,
        textposition=None,
        textfontsize=12,
        hoverinfo='none',
        showlegend=False
    ):
    """ 
    options for mode: lines, text, markers or any combination thereof, separated by commas
    options for hoverinfo: x, x+y, x+y+z, x+y+z+text, x+y+name, name, or any combination thereof
    options for textposition: top right, bottom left, middle left, middle center, bottom center, etc
    """
    if name is None: name = ''
    if text is None: text = ['']
    if textposition is None: textposition = ['top right'] * len(x)
    if textcolor is None: textcolor = color

    ld = go.Scatter3d(x = x, y = y, z = z,  
        mode = mode,
        name = name,
        line=dict(
            color=color,
            width=width,
        ),  
        text = text,
        textposition = textposition,
        textfont=dict(
            size=textfontsize,
            color=textcolor
        ),  
        hoverinfo = hoverinfo,
        showlegend = False
    )   
    return ld

def Scatter_R2(x, y,
        mode='lines',
        name=None,
        color='rgb(255,0,0)',
        width=3.,
        text=None,
        textcolor=None,
        textposition=None,
        textfontsize=12,
        hoverinfo='none',
        showlegend=False
    ):  
    """ 
    options for mode: lines, text, markers or any combination thereof, separated by commas
    options for hoverinfo: x, x+y, x+y+text, x+y+name, name, or any combination thereof
    options for textposition: top right, bottom left, middle left, middle center, bottom center, etc
    """
    if name is None: name = ''
    if text is None: text = ['']
    if textposition is None: textposition = ['top right'] * len(x)
    if textcolor is None: textcolor = color

    ld = go.Scatter(x = x, y = y,
        mode = mode,
        name = name,
        line=dict(
            color=color,
            width=width,
        ),  
        text = text,
        textposition = textposition,
        textfont=dict(
            size=textfontsize,
            color=textcolor
        ),  
        hoverinfo = hoverinfo,
        showlegend = showlegend
    )   
    return ld

py_text_sub = lambda txt, sub: r'{}<sub>{}</sub>'.format(txt, sub)
