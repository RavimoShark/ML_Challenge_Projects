"""Spectral clustering analysis."""
import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import BallTree
from scipy.sparse import coo_matrix

from kmeans_sol import kmeans


def compute_affinity_matrix(X, epsilon=None):
    """Compute the affinity matrix from the number of neighbors.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    epsilon: float | None.
        epsilon parameter to define the neighborhood of each sample.
        If None, it is infered from the data.

    Returns
    -------
    affinity_matrix: array-like, shape (n_samples, n_samples)
        Affinity matrix.
    """
    n_samples, _ = X.shape

    if epsilon is None:
        epsilon = 3. * np.mean(BallTree(X).query(X, 10)[0][:, -1])

    ind, d = BallTree(X).query_radius(X, r=epsilon, return_distance=True)
    data_pos = np.array([[value, r, c]
                        for r, (ind_r, data_r) in enumerate(zip(ind, d))
                        for c, value in zip(ind_r, data_r)])
    data_pos[:, 0] = np.exp(-.5 * (data_pos[:, 0] / epsilon) ** 2)

    W = coo_matrix((data_pos[:, 0], (data_pos[:, 1], data_pos[:, 2])),
                   shape=(n_samples, n_samples), dtype=np.float)
    W = W.toarray()
    return W


def spectral_clustering(X, n_clusters=2):
    """Compute the affinity matrix from the number of neighbors.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster: int, defaults to 2
        The number of clusters to form.

    Returns
    -------
    labels: array-like, shape (n_samples,)
        The estimated labels
    """
    # Q10: Complete the spectral clustering here.
    W = compute_affinity_matrix(X)

    L = np.diag(W.sum(1)) - W

    U = scipy.linalg.eigh(L)[1][:, :n_clusters]
    labels, _, _ = kmeans(U, n_clusters=n_clusters)
    return labels


if __name__ == '__main__':
    # Parameters
    random_state = 0
    n_samples, n_clusters, n_neighbors = 1000, 3, 10
    color = 'rgbcmyk'

    # Data to analyse
    datasets = {
        'blobs': make_blobs(n_samples=n_samples, random_state=random_state,
                            centers=3),
        'moons': make_moons(n_samples=n_samples, noise=.05, shuffle=True,
                            random_state=random_state),
        'circles': make_circles(n_samples=n_samples, factor=.5, noise=.05,
                                shuffle=True, random_state=random_state)
    }

    # Q9 - Q11 : Analysis of datasets
    plt.figure(figsize=(12, 8))

    for i, (_, data) in enumerate(datasets.items()):
        X, y = data
        n_clusters = np.max(y) + 1

        # K-Means
        t0 = time.time()
        labels_kmeans, _, _ = kmeans(X, n_clusters=n_clusters)
        time_kmeans = time.time() - t0

        # Spectral
        t0 = time.time()
        labels_spectral = spectral_clustering(X, n_clusters=n_clusters)
        time_spectral = time.time() - t0

        for j, (labels, t) in enumerate(zip((labels_kmeans, labels_spectral),
                                            (time_kmeans, time_spectral))):
            ax = plt.subplot(2, 3, 3 * j + i + 1)
            for k in range(n_clusters):
                ax.scatter(X[labels == k, 0], X[labels == k, 1],
                           color=color[k])

            ax.axis('equal')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.text(.99, .01, ('%.4fs' % t).lstrip('0'),
                    transform=plt.gca().transAxes, size=15,
                    horizontalalignment='right')
            if not i:
                ax.set_ylabel(['K-Means', 'Spectral Clustering'][j])

    plt.tight_layout()
    plt.show()
