import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from kmeans_sol import kmeans


def compute_log_inertia(X, n_clusters, T, bb_min, bb_max,
                        random_state=0):
    """Compute the log inertia of X and X_t.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_clusters: int
        The desired number of clusters.

    T: int
        Number of draws of X_t.

    bb_min: array, shape (n_features,)
        Inferior corner of the bounding box of X.

    bb_max: array, shape (n_features,)
        Superior corner of the bounding box of X.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    log_inertia: float
        Log of the inertia of the K-means applied to X.

    mean_log_inertia_rand: float
        Mean of the log of the inertia of the K-means applied to the different
        X_t.

    std_log_inertia_rand: float
        Standard deviation of the log of the inertia of the K-means applied to
        the different X_t.
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)

    # Compute inertia for real data
    _, _, inertia = kmeans(X, n_clusters=n_clusters)

    # Compute the random inertia
    rand_inertia = np.empty(T)
    for t in range(T):
        X_t = (rng.uniform(size=X.shape) * (bb_max - bb_min) + bb_min)
        _, _, rand_inertia[t] = kmeans(X_t, n_clusters=n_clusters)
    rand_inertia = np.log(rand_inertia)

    return np.log(inertia), np.mean(rand_inertia), np.std(rand_inertia)


def compute_gap(X, n_clusters_max, T=10, random_state=0):
    """Compute values of Gap and delta.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster_max: int
        Maximum number of cluster to test.

    T: int, defaults 10.
        Number of draws of X_t.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    n_clusters_range: array-like, shape(n_clusters_max-1,)
        Array of number of clusters tested.

    gap: array-like, shape(n_clusters_max-1,)
        Return the gap values.

    delta: array-like, shape(n_clusters_max-1,)
        Return the delta values.
    """
    n_sample, n_features = X.shape

    n_clusters_range = np.arange(1, n_clusters_max)
    bb_min, bb_max = np.min(X, 0), np.max(X, 0)

    log_inertia = np.empty(len(n_clusters_range))
    log_inertia_rand = np.empty(len(n_clusters_range))
    safety = np.empty(len(n_clusters_range))
    for k, n_clusters in enumerate(n_clusters_range):
        log_inertia[k], log_inertia_rand[k], safety[k] = compute_log_inertia(
            X, n_clusters, T, bb_min, bb_max)

    gap = log_inertia_rand - log_inertia
    delta = gap[:-1] - (gap[1:] - np.sqrt(1. + 1. / T) * safety[1:])

    return n_clusters_range, gap, delta


def plot_result(n_clusters_range, gap, delta):
    """Plot the values of Gap and delta.

    Parameters
    ----------
    n_clusters_range: array-like, shape(n_clusters_max-1,)
        Array of number of clusters tested.

    gap: array-like, shape(n_clusters_max-1,)
        Return the gap values.

    delta: array-like, shape(n_clusters_max-1,)
        Return the delta values.
    """
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=.05, right=.98, bottom=.08, top=.98, wspace=.15,
                        hspace=.03)

    plt.subplot(121)
    plt.plot(n_clusters_range, gap)
    plt.ylabel(r'$Gap(k)$', fontsize=18)
    plt.xlabel("Number of clusters")

    plt.subplot(122)
    for x, y in zip(n_clusters_range, delta):
        plt.bar(x - .45, y, width=0.9)
    plt.ylabel(r'$\delta(k)$', fontsize=18)
    plt.xlabel("Number of clusters")

    plt.draw()


def optimal_n_clusters_search(X, n_clusters_max, T=10, random_state=0):
    """Compute the optimal number of clusters.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster_max: int
        Maximum number of cluster to test.

    T: int, defaults 10.
        Number of draws of X_t.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    n_clusters_optimal: int
        Optimal number of clusters.
    """
    clusters_range, _, delta = compute_gap(X, n_clusters_max, T,
                                           random_state=0)
    for k, value in enumerate(delta):
        if value > 0:
            break

    return clusters_range[k]


if __name__ == '__main__':
    # Parameters
    random_state = 0
    n_samples, n_clusters_max = 1000, 10
    color = 'rgbcmyk'

    n_clusters = 5
    X, labels = make_blobs(n_samples=n_samples, random_state=random_state,
                           centers=n_clusters)

    plot_result(*compute_gap(X, n_clusters_max))

    plt.figure()
    n_clusters_opt = optimal_n_clusters_search(X, n_clusters_max)
    labels, _, _ = kmeans(X, n_clusters_opt)
    for k in range(n_clusters):
        plt.scatter(X[labels == k, 0], X[labels == k, 1], color=color[k])
        plt.axis("equal")
    plt.show()
