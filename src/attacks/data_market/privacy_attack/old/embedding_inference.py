import numpy as np
from numpy.linalg import eig
from sklearn.decomposition import PCA


def construct_fisher_information_matrix(selected_points, weights=None):
    """
    Constructs the Fisher Information Matrix from selected data points.

    Parameters:
    - selected_points (numpy.ndarray): Matrix of selected data points, shape (n_selected, n_features).
    - weights (numpy.ndarray): Weights associated with each selected point, shape (n_selected,).

    Returns:
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).
    """
    n_features = selected_points.shape[1]

    if weights is None:
        weights = np.ones(selected_points.shape[0])

    fisher_info_matrix = np.zeros((n_features, n_features))

    # Calculate the Fisher Information Matrix as a weighted sum of outer products
    for i, x in enumerate(selected_points):
        fisher_info_matrix += weights[i] * np.outer(x, x)

    return fisher_info_matrix


def perform_eigen_decomposition(fisher_info_matrix):
    """
    Perform eigen decomposition on the Fisher Information Matrix to find principal components.

    Parameters:
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).

    Returns:
    - eigenvalues (numpy.ndarray): Eigenvalues sorted in descending order.
    - eigenvectors (numpy.ndarray): Corresponding eigenvectors sorted by eigenvalue, shape (n_features, n_features).
    """
    eigenvalues, eigenvectors = eig(fisher_info_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def project_test_points(test_points, eigenvectors, n_components=2):
    """
    Project test points onto the principal components of the Fisher Information Matrix.

    Parameters:
    - test_points (numpy.ndarray): Matrix of test data points, shape (n_test, n_features).
    - eigenvectors (numpy.ndarray): Eigenvectors of Fisher Information Matrix, shape (n_features, n_features).
    - n_components (int): Number of top components to project onto.

    Returns:
    - projections (numpy.ndarray): Projected test points, shape (n_test, n_components).
    """
    principal_components = eigenvectors[:, :n_components]  # Select top eigenvectors
    projections = test_points @ principal_components  # Project test points

    return projections


import numpy as np


def reconstruct_embedding(fisher_info_matrix, n_components):
    """
    Reconstruct an embedding that best aligns with the given Fisher Information Matrix.

    Parameters:
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).
    - n_components (int): Number of top components to use in the reconstruction.

    Returns:
    - reconstructed_embedding (numpy.ndarray): The reconstructed embedding, shape (n_features,).
    """
    # Eigen decomposition of the Fisher Information Matrix
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_info_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top components and compute the reconstructed embedding
    top_eigenvectors = eigenvectors[:, :n_components]
    top_eigenvalues = eigenvalues[:n_components]

    # Weighted sum of principal components
    reconstructed_embedding = sum(np.sqrt(top_eigenvalues[i]) * top_eigenvectors[:, i] for i in range(n_components))

    # Normalize the reconstructed embedding (optional)
    reconstructed_embedding /= np.linalg.norm(reconstructed_embedding)

    return reconstructed_embedding


# Example usage
fisher_info_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])  # Example Fisher Information Matrix
n_components = 2  # Use top 2 components for reconstruction

reconstructed_embedding = reconstruct_embedding(fisher_info_matrix, n_components)
print("Reconstructed Embedding:\n", reconstructed_embedding)

# Example usage
# Assume we have the following data:
selected_points = np.array(
    [[1.2, 0.9], [0.8, 1.0], [1.5, 1.3], [0.7, 0.8]])  # Selected data points (n_selected, n_features)
weights = np.array([0.5, 0.8, 0.6, 0.7])  # Weights for each selected point
test_points = np.array([[1.0, 0.9], [0.5, 0.6], [1.4, 1.2]])  # New test data points to check alignment

# Step 1: Construct the Fisher Information Matrix
fisher_info_matrix = construct_fisher_information_matrix(selected_points, weights)

# Step 2: Perform eigen decomposition to get principal components
eigenvalues, eigenvectors = perform_eigen_decomposition(fisher_info_matrix)

# Step 3: Project test points onto the principal components
projections = project_test_points(test_points, eigenvectors, n_components=2)

print("Fisher Information Matrix:\n", fisher_info_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Principal Components (Eigenvectors):\n", eigenvectors)
print("Projections of Test Points:\n", projections)

import numpy as np


def reconstruct_embedding(fisher_info_matrix, n_components):
    """
    Reconstruct an embedding that best aligns with the given Fisher Information Matrix.

    Parameters:
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).
    - n_components (int): Number of top components to use in the reconstruction.

    Returns:
    - reconstructed_embedding (numpy.ndarray): The reconstructed embedding, shape (n_features,).
    """
    # Eigen decomposition of the Fisher Information Matrix
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_info_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top components and compute the reconstructed embedding
    top_eigenvectors = eigenvectors[:, :n_components]
    top_eigenvalues = eigenvalues[:n_components]

    # Weighted sum of principal components
    reconstructed_embedding = sum(np.sqrt(top_eigenvalues[i]) * top_eigenvectors[:, i] for i in range(n_components))

    # Normalize the reconstructed embedding (optional)
    reconstructed_embedding /= np.linalg.norm(reconstructed_embedding)

    return reconstructed_embedding


# Example usage
fisher_info_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])  # Example Fisher Information Matrix
n_components = 2  # Use top 2 components for reconstruction

reconstructed_embedding = reconstruct_embedding(fisher_info_matrix, n_components)
print("Reconstructed Embedding:\n", reconstructed_embedding)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans


def analyze_unselected_points(selected_points, unselected_points, fisher_info_matrix, n_components=2):
    """
    Analyze unselected points in relation to selected points using Fisher Information Matrix.

    Parameters:
    - selected_points (numpy.ndarray): Matrix of selected data points, shape (n_selected, n_features).
    - unselected_points (numpy.ndarray): Matrix of unselected data points, shape (n_unselected, n_features).
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).
    - n_components (int): Number of top components to use in the analysis.

    Returns:
    - unselected_distances (numpy.ndarray): Distances of unselected points from selected regions in feature space.
    """
    # Perform eigen decomposition to get principal components
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_info_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Project selected and unselected points onto principal components
    selected_projections = selected_points @ top_eigenvectors
    unselected_projections = unselected_points @ top_eigenvectors

    # Calculate distance of each unselected point to the centroid of selected points
    selected_centroid = np.mean(selected_projections, axis=0)
    unselected_distances = euclidean_distances(unselected_projections, selected_centroid.reshape(1, -1)).flatten()

    # Optional: Clustering to identify boundaries
    all_projections = np.vstack((selected_projections, unselected_projections))
    kmeans = KMeans(n_clusters=2).fit(all_projections)
    labels = kmeans.labels_

    return unselected_distances, labels


# Example usage
selected_points = np.random.rand(5, 3)  # Example selected data points
unselected_points = np.random.rand(8, 3)  # Example unselected data points
fisher_info_matrix = np.cov(selected_points, rowvar=False)  # Example Fisher Information Matrix

unselected_distances, cluster_labels = analyze_unselected_points(selected_points, unselected_points, fisher_info_matrix)
print("Distances of Unselected Points to Selected Region:\n", unselected_distances)
print("Cluster Labels for Combined Points:\n", cluster_labels)

import numpy as np
from numpy.linalg import eigh


def construct_fisher_matrix(data_points, weights=None):
    """
    Construct a Fisher Information Matrix for a given set of data points.

    Parameters:
    - data_points (numpy.ndarray): Array of data points, shape (n_points, n_features).
    - weights (numpy.ndarray): Weights for each data point, shape (n_points,).

    Returns:
    - fisher_matrix (numpy.ndarray): Fisher Information Matrix, shape (n_features, n_features).
    """
    if weights is None:
        weights = np.ones(data_points.shape[0])

    fisher_matrix = np.zeros((data_points.shape[1], data_points.shape[1]))
    for i, x in enumerate(data_points):
        fisher_matrix += weights[i] * np.outer(x, x)

    return fisher_matrix


def refine_with_combined_fisher(selected_points, unselected_points, alpha=1.0, n_components=2):
    """
    Construct a refined Fisher Information Matrix using both selected and unselected points.

    Parameters:
    - selected_points (numpy.ndarray): Selected data points, shape (n_selected, n_features).
    - unselected_points (numpy.ndarray): Unselected data points, shape (n_unselected, n_features).
    - alpha (float): Scaling factor for unselected points' influence.
    - n_components (int): Number of top components to use for the refined embedding.

    Returns:
    - refined_embedding (numpy.ndarray): Refined reconstructed embedding, shape (n_features,).
    """
    # Construct Fisher Information Matrices for selected and unselected points
    fisher_selected = construct_fisher_matrix(selected_points)
    fisher_unselected = construct_fisher_matrix(unselected_points)

    # Construct the combined Fisher Information Matrix
    fisher_combined = fisher_selected - alpha * fisher_unselected

    # Perform eigen decomposition
    eigenvalues, eigenvectors = eigh(fisher_combined)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvalues = eigenvalues[sorted_indices[:n_components]]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Construct the refined embedding using the top components
    refined_embedding = sum(np.sqrt(top_eigenvalues[i]) * top_eigenvectors[:, i] for i in range(n_components))

    # Normalize the refined embedding (optional)
    refined_embedding /= np.linalg.norm(refined_embedding)

    return refined_embedding


# Example usage
selected_points = np.random.rand(5, 3)  # Example selected points
unselected_points = np.random.rand(8, 3)  # Example unselected points

# Construct the refined embedding
refined_embedding = refine_with_combined_fisher(selected_points, unselected_points, alpha=1.0, n_components=2)
print("Refined Embedding:\n", refined_embedding)

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def sample_near_miss_points(selected_points, unselected_points, threshold=0.5, max_samples=10):
    """
    Samples unselected points that are similar to selected points based on distance.

    Parameters:
    - selected_points (numpy.ndarray): Selected data points, shape (n_selected, n_features).
    - unselected_points (numpy.ndarray): Unselected data points, shape (n_unselected, n_features).
    - threshold (float): Distance threshold for similarity.
    - max_samples (int): Maximum number of near-miss points to sample.

    Returns:
    - sampled_near_miss_points (numpy.ndarray): Sampled near-miss unselected points, shape (n_samples, n_features).
    """
    # Compute centroid of selected points
    selected_centroid = np.mean(selected_points, axis=0)

    # Calculate distances from each unselected point to the selected centroid
    distances = euclidean_distances(unselected_points, selected_centroid.reshape(1, -1)).flatten()

    # Select points within the distance threshold
    near_miss_indices = np.where(distances <= threshold)[0]
    sampled_indices = np.random.choice(near_miss_indices, min(max_samples, len(near_miss_indices)), replace=False)

    return unselected_points[sampled_indices]


# Example usage
selected_points = np.random.rand(5, 3)  # Example selected points
unselected_points = np.random.rand(15, 3)  # Example unselected points

# Sample near-miss points
sampled_near_miss_points = sample_near_miss_points(selected_points, unselected_points, threshold=0.3, max_samples=5)
print("Sampled Near-Miss Unselected Points:\n", sampled_near_miss_points)

import numpy as np


def sample_embeddings_from_space(fisher_info_matrix, n_components=2, n_samples=10, random_scale=1.0):
    """
    Sample random embeddings from the space defined by the top components of the Fisher Information Matrix.

    Parameters:
    - fisher_info_matrix (numpy.ndarray): The Fisher Information Matrix, shape (n_features, n_features).
    - n_components (int): Number of top components to use for the embedding space.
    - n_samples (int): Number of random embeddings to sample.
    - random_scale (float): Scale for random coefficients.

    Returns:
    - sampled_embeddings (numpy.ndarray): Sampled embeddings, shape (n_samples, n_features).
    """
    # Perform eigen decomposition to get top components
    eigenvalues, eigenvectors = np.linalg.eigh(fisher_info_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    top_eigenvalues = eigenvalues[sorted_indices[:n_components]]

    # Sample random embeddings as linear combinations of top components
    sampled_embeddings = []
    for _ in range(n_samples):
        random_coefficients = np.random.normal(loc=0, scale=random_scale, size=n_components) * np.sqrt(top_eigenvalues)
        sampled_embedding = top_eigenvectors @ random_coefficients  # Linear combination
        sampled_embeddings.append(sampled_embedding)

    return np.array(sampled_embeddings)


# Example usage
fisher_info_matrix = np.random.rand(3, 3)  # Example Fisher Information Matrix
n_components = 2  # Top components to define embedding space
n_samples = 5  # Number of random embeddings to sample
sampled_embeddings = sample_embeddings_from_space(fisher_info_matrix, n_components, n_samples, random_scale=1.0)
print("Sampled Embeddings from Embedding Space:\n", sampled_embeddings)
