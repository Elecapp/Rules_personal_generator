"""
UMAP Dimensionality Reduction Utilities

This module provides utilities for UMAP (Uniform Manifold Approximation and Projection)
dimensionality reduction and visualization. It's used to create 2D projections of
high-dimensional feature spaces for visualization of neighborhoods and decision boundaries.

UMAP is particularly useful for:
- Visualizing how synthetic neighborhoods relate to training data
- Understanding feature space structure
- Identifying cluster patterns in generated data

Functions:
    run_umap: Apply UMAP to reduce dimensionality
    grid_search_umap: Test multiple UMAP parameter combinations

The module uses Altair for visualization and supports parameter grid search
to find optimal UMAP configurations for different datasets.
"""

import umap
import umap.umap_ as umap
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.enable('default', max_rows=None)


def run_umap(data, labels, n_neighbors=5, min_dist=0.1, n_components=2, metric='euclidean'):
    """
    Apply UMAP dimensionality reduction to data.
    
    Creates a low-dimensional embedding (typically 2D) of high-dimensional data
    using UMAP algorithm. This is useful for visualization and understanding
    the structure of feature spaces.
    
    Args:
        data: High-dimensional data array (n_samples, n_features)
        labels: Class labels for each sample (used for visualization)
        n_neighbors: Number of neighbors for UMAP (default 5)
                    - Smaller values preserve local structure
                    - Larger values preserve global structure
        min_dist: Minimum distance between points in embedding (default 0.1)
                 - Smaller values create tighter clusters
                 - Larger values preserve more global structure
        n_components: Number of dimensions in embedding (default 2)
        metric: Distance metric to use (default 'euclidean')
               - Options: 'euclidean', 'manhattan', 'chebyshev', 'cosine', etc.
    
    Returns:
        Array: Low-dimensional embedding of the data (n_samples, n_components)
    """
    reducer= umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    embedding = reducer.fit_transform(data)
    return embedding


def grid_search_umap(X_train, neigh_arrays, labels, min_dists, metrics): #n_neighbors_values
    """
    Perform grid search over UMAP parameters for multiple datasets.
    
    This function helps find optimal UMAP parameters by testing multiple
    combinations of min_dist and metric on combined training and neighborhood data.
    It's particularly useful for visualizing how different neighborhoods compare
    to the training data.
    
    The function combines multiple datasets (training + neighborhoods), applies
    UMAP with different parameter combinations, and returns results in a format
    suitable for visualization with Altair.
    
    Args:
        X_train: Training data array
        neigh_arrays: List of neighborhood data arrays to compare
        labels: Optional list of dataset names for each array
                If None, uses generic names 'Dataset 0', 'Dataset 1', etc.
        min_dists: List of min_dist values to test
        metrics: List of distance metrics to test
    
    Returns:
        DataFrame: Results with columns ['x', 'y', 'label', 'metric', 'min_dist']
                  - x, y: UMAP embedding coordinates
                  - label: Dataset identifier (0 for training, 1+ for neighborhoods)
                  - metric: Distance metric used
                  - min_dist: Min distance parameter used
                  
    Example:
        >>> results = grid_search_umap(
        ...     X_train, 
        ...     [random_neigh, genetic_neigh], 
        ...     ['Train', 'Random', 'Genetic'],
        ...     [0.1, 0.3],
        ...     ['euclidean', 'chebyshev']
        ... )
    """
    if labels is None:
        labels = ['Dataset {}'.format(i) for i in range(len(neigh_arrays) + 1)]

    combined_data = [X_train] + list(neigh_arrays)
    data = np.vstack(combined_data)
    label_array = np.concatenate([[i] * len(d) for i, d in enumerate(combined_data)])

    grid_results = []

    for min_dist in min_dists:
        for metric in metrics:
        #for n_neighbors in n_neighbors_values:
            embedding = run_umap(data, label_array, metric=metric , min_dist=min_dist) #n_neighbors=n_neighbors
            for i, (x, y) in enumerate(embedding):
                grid_results.append({
                    'x': x,
                    'y': y,
                    'label': label_array[i],
                    'metric':metric,
                    #'n_neighbors': n_neighbors,
                    'min_dist': min_dist
                })
    return pd.DataFrame(grid_results)


def visualize_with_altair(grid_results):
    label_mapping = {
        0: "Train",
        1: "Random",
        2: "Custom",
        3: "Genetic",
        4: "Custom Genetic"
    }
    grid_results['label_name'] =grid_results['label'].map(label_mapping)
    domain_ = ['Train', 'Random', 'Custom', 'Genetic', 'Custom Genetic']
    range_ = ['#909396', '#f08a24','#1ceb7c','#ffe600','#1805f0']
    chart = alt.Chart(grid_results).mark_circle(size=25, opacity=0.3).encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('label_name:N', legend=alt.Legend(title='Labels'), scale=alt.Scale(domain= domain_, range=range_) ),
        tooltip=['x','y']
    ).facet(
        column=alt.Column('metric:N', title='Metric'),
        row=alt.Row('min_dist:Q', title='min_dist')
    ).properties(
        title="UMAP Grid Search Visualization"
    )
    alt.renderers.enable("browser")
    return chart

def save_umap_to_csv(grid_results, output_csv_path):
    """
    Save the UMAP embeddings along with their labels, metrics, and parameters into a CSV file.

    Args:
        grid_results (pd.DataFrame): DataFrame containing UMAP results.
        output_csv_path (str): Path to save the resulting CSV file.
    """
    # Save the DataFrame to a CSV file
    grid_results.to_csv(output_csv_path, index=False)
    print(f"UMAP embeddings exported to {output_csv_path}")

if __name__ == '__main__':
    output_dir = 'umap_n_neighs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_feat = pd.read_csv("precomputed_csv/X_feat.csv")
    random_n = pd.read_csv("precomputed_csv/random_n.csv")
    custom_n = pd.read_csv("precomputed_csv/custom_n.csv")
    genetic_n = pd.read_csv("precomputed_csv/genetic_n.csv")
    custom_genetic_n = pd.read_csv("precomputed_csv/custom_genetic_n.csv")
    X_feat_array = X_feat.to_numpy()
    random_n_array = random_n.to_numpy()
    custom_n_array = custom_n.to_numpy()
    genetic_n_array = genetic_n.to_numpy()
    custom_genetic_n_array = custom_genetic_n.to_numpy()

    min_dist_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    #n_neighbors_values = [5, 6, 7, 8, 9]
    metric_values = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    grid_results = grid_search_umap(
        X_feat_array, [random_n_array, custom_n_array, genetic_n_array, custom_genetic_n_array],
        labels=["Train", "Random", "Custom", "Genetic", "Custom Genetic"],
        min_dists=min_dist_values,
        #n_neighbors_values=n_neighbors_values,
        metrics = metric_values
    )

    output_csv_path = os.path.join(output_dir, "umap_embeddings.csv")
    save_umap_to_csv(grid_results, output_csv_path)

    # Visualize results using Altair
    chart = visualize_with_altair(grid_results)
    chart.show()


