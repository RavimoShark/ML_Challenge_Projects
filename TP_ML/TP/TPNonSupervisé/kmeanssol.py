"""Implemantation of the Kmeans algorithm."""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial

from sklearn.datasets import make_blobs
import seaborn as sns


def compute_labels(X, centroids):
    """Compute labels.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.

    Returns
    -------
    labels : array, shape (n_samples,)
        The labels of each sample
    """
    dist = spatial.distance.cdist(X, centroids, metric='euclidean')
    return dist.argmin(axis=1)


def compute_inertia_centroids(X, labels):
    """Compute inertia and centroids.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like, shape (n_saples,)
        The labels of each sample.

    Returns
    -------
    inertia: float
        The inertia.

    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.
    """
    labels_unique = np.unique(labels)
    centroids = np.empty((len(labels_unique), X.shape[1]), dtype=X.dtype)
    inertia = 0.
    for k, l in enumerate(labels_unique):
        X_l = X[labels == l]
        centroids[k] = np.mean(X_l, axis=0)
        dist = spatial.distance.cdist(X_l, [centroids[k]], metric='euclidean')
        inertia += np.sum(dist ** 2)
    return inertia, centroids


def kmeans(X, n_clusters, n_iter=100, tol=1e-7, random_state=42):
    """Estimate position of centroids and labels.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_clusters: int
        The desired number of clusters.

    max_iter: int, defaults 100.
        Max number of update.

    tol: float, defaults 1e-7.
        The tolerance to check convergence.

    random_state: int, defaults to 42.
        A random number generator instance.

    Returns
    -------
    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.

    labels: array-like, shape (n_samples,)
        The estimated labels.

    inertia: float
        The inertia.
    """
    # initialize centroids with random samples
    rng = np.random.RandomState(random_state)
    centroids = X[rng.permutation(len(X))[:n_clusters]]

    labels = compute_labels(X, centroids)
    old_inertia = np.inf
    for _ in range(n_iter):
        inertia, centroids = compute_inertia_centroids(X, labels)
        if abs(inertia - old_inertia) < tol:
            break
        old_inertia = inertia
        labels = compute_labels(X, centroids)
    else:
        warnings.warn("The algorithm did not converge.")

    return labels, centroids, inertia


if __name__ == '__main__':
    # Parameters
    random_state = 0
    n_samples = 1000
    color = 'rgbcmyk'

    # Generate data
    X, y = make_blobs(n_samples=n_samples, random_state=random_state,
                      centers=3)

    # Q1-Q4 Apply K-means to X
    plt.figure()
    n_clusters_max = 6

    for n_clusters in range(2, n_clusters_max + 2):
        labels, _, _ = kmeans(X, n_clusters=n_clusters)

        plt.subplot(3, 2, n_clusters - 1)
        plt.title("Number of cluster = %d" % n_clusters)
        for k in range(n_clusters):
            plt.scatter(X[labels == k, 0], X[labels == k, 1], color=color[k])
            plt.axis("equal")

    plt.tight_layout()
    plt.show()
