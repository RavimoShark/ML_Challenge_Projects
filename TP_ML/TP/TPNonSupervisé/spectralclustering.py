"""Spectral clustering analysis."""
import numpy as np

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import BallTree
from scipy.sparse import coo_matrix


def compute_affinity_matrix(X, epsilon=None):
    """Compute the affinity matrix from the number of neighbors.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    epsilon: float | None.
        epsilon parameter to define the neighborhood of each sample.
        If None, it is inferred from the data.

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


def spectral_clustering(X, n_clusters=2, n_neighbors=10):
    """Compute the affinity matrix from the number of neighbors.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster: int, defaults to 2
        The number of clusters to form.

    n_neighbors: int, defaults to 10
        Number of neighbors considered to compute the affinity matrix.

    Returns
    -------
    labels: array-like, shape (n_samples,)
        The estimated labels
    """
    # Q10: Complete the spectral clustering here.
    pass


if __name__ == '__main__':
    # Parameters
    random_state = 0
    n_samples, n_clusters, n_neighbors = 1000, 3, 10

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
