# Intel®-oneAPI

#### Team Name - Alpha Tribe
#### Problem Statement - Open Innovation in Education
#### Team Leader Email - subhajitmajiofficial@gmail.com

## A Brief of the Prototype:
  Fractal equations are complex but beautiful to watch and explore when plotted on a 3d graph by creating plotting points through numerous iterations.
  That's what we explored in this project. We computed and plotted various fractal equations using a series of libraries and Intel® OneAPI Toolkit to optimize the process.
  
  You can learn about the project in details from my medium blog - https://medium.com/@subhajitmajiofficial/intel-oneapi-hackathon-submission-40543e01d6b6.
  
## Tech Stack: 
   1. NUMPY
   2. PLOTLY
   3. STREAMLIT
   4. SCIKIT LEARN
   5. INTEL® OneAPI Toolkit
   
## Step-by-Step Code Execution Instructions:
  We first imported all the necessary libraries such as numpy for computing the fractal equations,
  ```
  import numpy as np
  ```
  
  plotly for vizualizing the equations,
  ```
  import plotly.graph_objs as go
  ```
  
  streamlit for creating a web app
  ```
  import streamlit as st
  ```
  
  
  and finally scikit-learn extensions to take use of the accelerated computing and optimizations by patching it with Intel® OneAPI toolkit.
  ```
  from sklearnex import patch_sklearn
  from sklearnex.cluster import KMeans
  ```
  
  Next we define the fractal equations one by one..we covered a few of them in this prototype...we hope to explore as many of them as we can in the final developement phase.
  
  Mandelbrot Set,
  ```
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
   ```
    
   Mandelbrot Cubic,
   ```
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
   ```
   
   Mandelbrot Quartic,
   ```
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
   ```
    
   Burning Ship,
   ```
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
   ```
   
   Julia Set,
   ```
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
   ```
   
   Phoenix Set,
   ```
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
   ```
   
   and Tricorn.
   ```
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
   ```
  
Next we defined the 3d visualization function to generate a 3d scatter plot.
```
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
```

We then created a web app to test our program using Streamlit.
```
st.title('Fractal 3D Visualization')

fractals = {
    'Mandelbrot': mandelbrot,
    'Mandelbrot Cubic': mandelbrot_cubic,
    'Mandelbrot Quartic': mandelbrot_quartic,
    'Burning Ship': burning_ship,
    'Julia': julia,
    'Phoenix': phoenix,
    'Tricorn': tricorn,
}

fractal_name = st.selectbox('Select Fractal', list(fractals.keys()))

xmin, xmax = st.slider('Real Axis Range', -3.0, 3.0, (-2.0, 1.0), 0.1)
ymin, ymax = st.slider('Imaginary Axis Range', -3.0, 3.0, (-1.5, 1.5), 0.1)
resolution = st.slider('Resolution', 50, 500, 100, 50)

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
```    
  
## What I Learned:
   Fractals are never ending intricate patterns that are infinitely complex yet can be generated using simple algorithms. This is what we wanted to explore. We learnt that, although formulated through simple algorithms, the real result comes out when looping the algorithm through numerous iterations. The more the no of iterations, the more beautiful the pattern will be. But that also means using more computational power and thus a lot of time. We explored Intel®'s OneAPI toolkit to help us with this problem. By patching the scikit learn tools with Intel®'s OneAPI, we greatly optimized the visualization process. Currently we only tested upto a 1000 iterations( we included the 100 iterations in the prototype because it pretty much gives us a rough idea of what the final output will look). We aim to test upto 10000 iterations in the final developement phase.
