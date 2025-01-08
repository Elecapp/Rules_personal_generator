import umap
import umap.umap_ as umap
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def draw_umap(data,labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='',save_path=None):
    reducer= umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    embedding = reducer.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(embedding[:, 0], range(len(embedding)), c=labels, cmap='Spectral')
    elif n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='Spectral', s=100)
    plt.title(title, fontsize=18)
    plt.colorbar(scatter, ticks=np.unique(labels))
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')


def visualize_umap(X_train, *neigh_arrays, labels=None):
    if labels is None:
        labels = ['Dataset {}'.format(i) for i in range(len(neigh_arrays) + 1)]
    combined_data = [X_train] + list(neigh_arrays)
    data = np.vstack(combined_data)
    label_array = np.concatenate([[i] * len(d) for i, d in enumerate(combined_data)])
    for n in (5, 6, 7, 8, 9):
        save_path = os.path.join(output_dir, f'umap_n_neighbors_{n}.png')
        draw_umap(data, label_array, n_neighbors=n, title='UMAP with n_neighbors = {}'.format(n), save_path=save_path)

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

    # Visualize UMAP with specified labels
    visualize_umap(X_feat_array, random_n_array, custom_n_array, genetic_n_array, custom_genetic_n_array,
                   labels=["Train", "Random", "Custom", "Genetic", "Custom Genetic"])

