import streamlit as st
import plotly.express as px
import numpy as np
from src.kmeans import KMeans
from src.data_generator import DataGenerator

st.title("K-Means Clustering Visualization")

# 侧边栏参数
with st.sidebar:
    dataset = st.selectbox(
        "Select Dataset",
        ["Gaussian Mixture", "Circles"]
    )

    k = st.slider("Number of Clusters (K)", 2, 10, 3)
    max_iter = st.slider("Max Iterations", 1, 100, 20)

    init_method = st.selectbox(
        "Initialization Method",
        ["random", "k-means++"]
    )

# 生成数据
data_gen = DataGenerator()
if dataset == "Gaussian Mixture":
    X = data_gen.gaussian_mixture(n_samples=300, centers=3)
else:
    X = data_gen.circles(n_samples=300)

# 执行聚类
kmeans = KMeans(k=k, max_iter=max_iter, init=init_method)
states = kmeans.fit(X)

# 创建动画帧
frames = []
for state in states:
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'Cluster': state.labels
    })

    # 添加质心
    centroids_df = pd.DataFrame({
        'x': state.centroids[:, 0],
        'y': state.centroids[:, 1],
        'Cluster': range(k)
    })

    fig = px.scatter(df, x='x', y='y', color='Cluster',
                     title=f'Iteration {state.iteration}')

    # 添加质心标记
    fig.add_scatter(x=centroids_df['x'], y=centroids_df['y'],
                    mode='markers', marker_symbol='x',
                    marker_size=15, showlegend=False)

    frames.append(fig)

# 显示动画控制器
current_iter = st.slider("Iteration", 0, len(frames)-1, 0)
st.plotly_chart(frames[current_iter])

# 显示收敛曲线
inertias = [state.inertia for state in states]
fig_conv = px.line(y=inertias, title="Convergence Plot")
st.plotly_chart(fig_conv)
