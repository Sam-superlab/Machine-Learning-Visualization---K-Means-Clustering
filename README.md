# K-Means Clustering Visualization

## Project Overview

This project provides an interactive visualization of the K-Means clustering algorithm, allowing users to understand how the algorithm works through dynamic visual feedback. The application is built using Python, Streamlit, and Plotly, and it supports various initialization methods and datasets.

## Project Goals

1. **Dynamic Visualization**: Display the clustering process in real-time, showing how data points are reassigned to clusters and how centroids move during iterations.
2. **Interactive Controls**: Allow users to select different datasets, initialization methods, and the number of clusters (K) through an intuitive user interface.
3. **Multiple Initialization Strategies**: Support various centroid initialization strategies, including random initialization and K-Means++.
4. **Dataset Variety**: Test the K-Means algorithm on different datasets, such as Gaussian mixtures and circular distributions, to observe its performance and limitations.
5. **Convergence Analysis**: Provide visual feedback on the convergence of the algorithm by displaying the sum of squared errors (SSE) over iterations.

## Project Structure

```
kmeans_visualization/
├── src/
│   ├── data_generator.py      # Script for generating or loading datasets
│   └── kmeans.py              # Core implementation of the K-Means algorithm
├── app.py                     # Main application file for Streamlit
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## Installation

To run this project, you need to have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd kmeans_visualization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv kmeans_env
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     kmeans_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source kmeans_env/bin/activate
     ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

After running the command, you can access the application in your web browser at `http://localhost:8501`.

### Features

- **Select Dataset**: Choose between "Gaussian Mixture" and "Circles" datasets.
- **Number of Clusters (K)**: Adjust the number of clusters using a slider.
- **Max Iterations**: Set the maximum number of iterations for the K-Means algorithm.
- **Initialization Method**: Choose between "random" and "k-means++" initialization strategies.
- **Iteration Control**: Navigate through the iterations to see how the algorithm converges.

## Example Output

The application will display a scatter plot of the data points, color-coded by their assigned clusters, along with the centroids marked distinctly. Additionally, a convergence plot will show the SSE over iterations, providing insight into the algorithm's performance.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an easy way to create web applications for data science.
- [Plotly](https://plotly.com/python/) for interactive plotting capabilities.
- [NumPy](https://numpy.org/) for numerical operations.
