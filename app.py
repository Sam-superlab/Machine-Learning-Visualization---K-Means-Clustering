import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from src.kmeans import KMeans
from src.data_generator import DataGenerator

st.title("K-Means Clustering Visualization")

# Initialize session state for current iteration
if 'current_iter' not in st.session_state:
    st.session_state.current_iter = 0

# Sidebar parameters
with st.sidebar:
    dataset = st.selectbox(
        "Select Dataset",
        ["Gaussian Mixture", "Circles"]
    )

    k = st.slider("Number of Clusters (K)", 2, 10, 3)
    max_iter = st.slider("Max Iterations", 1, 100, 20)

    init_method = st.selectbox(
        "Initialization Method",
        ["random", "k-means++"]  # Restore k-means++ option
    )

# Generate data
data_gen = DataGenerator()
if dataset == "Gaussian Mixture":
    X = data_gen.gaussian_mixture(n_samples=300, centers=3)
else:
    X = data_gen.circles(n_samples=300)

# Execute clustering
kmeans = KMeans(k=k, max_iter=max_iter, init=init_method)
states = kmeans.fit(X)

# Create animation frames
frames = []
for state in states:
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'Cluster': state.labels
    })

    # Add centroids
    centroids_df = pd.DataFrame({
        'x': state.centroids[:, 0],
        'y': state.centroids[:, 1],
        'Cluster': range(k)
    })

    fig = px.scatter(df, x='x', y='y', color='Cluster',
                     title=f'Iteration {state.iteration}')

    # Add centroid markers
    fig.add_scatter(x=centroids_df['x'], y=centroids_df['y'],
                    mode='markers', marker_symbol='x',
                    marker_size=15, showlegend=False)

    frames.append(fig)

# Display animation controller
current_iter = st.slider("Iteration", 0, len(
    frames)-1, st.session_state.current_iter)
st.session_state.current_iter = current_iter  # Update session state

st.plotly_chart(frames[current_iter])

# Display convergence curve
inertias = [state.inertia for state in states]
fig_conv = px.line(y=inertias, title="Convergence Plot")
st.plotly_chart(fig_conv)
