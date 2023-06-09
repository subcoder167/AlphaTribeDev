import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearnex import patch_sklearn
from sklearnex.cluster import KMeans

# Patch scikit-learn with Intel optimizations
patch_sklearn()

# Define the fractal functions
def mandelbrot(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z**2 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def mandelbrot_cubic(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z**3 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def mandelbrot_quartic(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z**4 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def burning_ship(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = (abs(z.real) + abs(z.imag)*1j)**2 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def julia(c, max_iterations=100):
    z = c
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z**2 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def phoenix(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = np.sin(z.real)**2 - np.sin(z.imag)**2 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

def tricorn(c, max_iterations=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = np.conj(z)**2 + c
        n += 1
    if n == max_iterations:
        return max_iterations
    else:
        return n

# Define the 3D visualization function
def plot_fractal_3d(fractal_func, xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, resolution=100):
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            c = x + y*1j
            z_val = fractal_func(c)
            if not np.isnan(z_val):
                points.append([x, y, z_val])
    points = np.array(points)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
    labels = kmeans.labels_
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=labels,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='Real',
        yaxis_title='Imaginary',
        zaxis_title='Iterations'))

    return fig

# Create the Streamlit app
st.title('Fractal 3D Visualization')

# Define the fractals and corresponding labels
fractals = {
    'Mandelbrot': mandelbrot,
    'Mandelbrot Cubic': mandelbrot_cubic,
    'Mandelbrot Quartic': mandelbrot_quartic,
    'Burning Ship': burning_ship,
    'Julia': julia,
    'Phoenix': phoenix,
    'Tricorn': tricorn,
}

# Create the dropdown input for selecting the fractal
fractal_name = st.selectbox('Select Fractal', list(fractals.keys()))

xmin, xmax = st.slider('Real Axis Range', -3.0, 3.0, (-2.0, 1.0), 0.1)
ymin, ymax = st.slider('Imaginary Axis Range', -3.0, 3.0, (-1.5, 1.5), 0.1)
resolution = st.slider('Resolution', 50, 500, 100, 50)

# Display the formula for the selected fractal
st.subheader('Formula')
if fractal_name in fractals:
    if fractal_name == 'Mandelbrot':
        st.latex(r'z_{n+1} = z_n^2 + c')
    elif fractal_name == 'Mandelbrot Cubic':
        st.latex(r'z_{n+1} = z_n^3 + c')
    elif fractal_name == 'Mandelbrot Quartic':
        st.latex(r'z_{n+1} = z_n^4 + c')
    elif fractal_name == 'Burning Ship':
        st.latex(r'z_{n+1} = (|Re(z_n)| + |Im(z_n)|i)^2 + c')
    elif fractal_name == 'Julia':
        st.latex(r'z_{n+1} = z_n^2 + c')
    elif fractal_name == 'Phoenix':
        st.latex(r'z_{n+1} = \sin^2(Re(z_n)) - \sin^2(Im(z_n)) + c')
    elif fractal_name == 'Tricorn':
        st.latex(r'z_{n+1} = \bar{z_n}^2 + c')

fig = plot_fractal_3d(fractals[fractal_name], xmin, xmax, ymin, ymax, resolution)
st.plotly_chart(fig, use_container_width=True)
