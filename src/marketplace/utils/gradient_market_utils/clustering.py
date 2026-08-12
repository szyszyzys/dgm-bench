import kmeans1d
import numpy as np
from sklearn.cluster import KMeans


def kmeans(x, k):
    clusters, centroids = kmeans1d.cluster(x, k)
    return clusters, centroids


def optimal_k_gap(X, k_max=5, B=10, random_state=42):
    """
    Compute the optimal number of clusters for data X using the Gap Statistic.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data to be clustered.
    k_max : int, default=5
        Maximum number of clusters to evaluate.
    B : int, default=10
        Number of reference datasets to generate.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    optimal_k : int
        The estimated optimal number of clusters.
    """
    # Ensure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    np.random.seed(random_state)
    n_samples, n_features = X.shape
    ks = np.arange(1, k_max + 1)

    epsilon = 1e-10  # small value to avoid log(0)

    def compute_dispersion(data, k):
        """
        Compute within-cluster dispersion for data given k clusters.
        """
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        dispersion = 0.0
        for j in range(k):
            cluster_data = data[kmeans.labels_ == j]
            if cluster_data.shape[0] > 0:
                center = np.mean(cluster_data, axis=0)
                dispersion += np.sum((cluster_data - center) ** 2)
        # Ensure dispersion is not zero
        if dispersion < epsilon:
            dispersion = epsilon
        return dispersion

    # Compute dispersion for the actual data for each k
    Wks = np.array([compute_dispersion(X, k) for k in ks])

    # Generate B reference datasets and compute dispersions for each k
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    Wkbs = np.zeros((len(ks), B))
    for b in range(B):
        X_ref = np.random.uniform(low=mins, high=maxs, size=(n_samples, n_features))
        for i, k in enumerate(ks):
            Wkbs[i, b] = compute_dispersion(X_ref, k)

    logWks = np.log(Wks + epsilon)
    logWkbs = np.log(Wkbs + epsilon)
    gap = np.mean(logWkbs, axis=1) - logWks
    sdk = np.std(logWkbs, axis=1) * np.sqrt(1 + 1.0 / B)

    optimal_k = ks[-1]
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sdk[i + 1]:
            optimal_k = ks[i]
            break

    return optimal_k
