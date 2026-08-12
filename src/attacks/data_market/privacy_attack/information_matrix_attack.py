from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class SelectionRound:
    """Information from one round of DAVED selection"""
    info_matrix: np.ndarray  # Current information matrix
    selected_point: np.ndarray  # Selected data point
    step_size: float  # αt in the paper
    previous_matrix: np.ndarray  # Previous information matrix


class InfoMatrixAttack:
    """Attack based on information matrix updates in DAVED"""

    def __init__(self,
                 seller_embeddings: np.ndarray,
                 embedding_dim: int,
                 n_clusters: int = 5):
        """
        Args:
            seller_embeddings: Available embeddings from seller
            embedding_dim: Dimension of embeddings
            n_clusters: Number of clusters for analysis
        """
        self.seller_embeddings = seller_embeddings
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters

        # Initialize structures for tracking
        self.matrix_changes = []
        self.principal_directions = []
        self.influence_scores = np.zeros(len(seller_embeddings))

    def analyze_matrix_update(self, round_info: SelectionRound) -> Dict:
        """Analyzes a single round of information matrix update"""

        # Calculate matrix difference
        matrix_diff = round_info.info_matrix - round_info.previous_matrix

        # Compute eigendecomposition of the difference
        eigenvals, eigenvecs = np.linalg.eigh(matrix_diff)

        # Sort by absolute eigenvalue
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Store principal directions
        self.principal_directions.append(eigenvecs[:, 0])  # Most significant direction

        # Calculate influence on seller points
        influences = np.abs(self.seller_embeddings @ eigenvecs[:, 0])
        self.influence_scores += influences

        return {
            'principal_direction': eigenvecs[:, 0],
            'eigenvalues': eigenvals,
            'point_influences': influences
        }

    def analyze_selection_sequence(self,
                                   selection_rounds: List[SelectionRound]) -> Dict:
        """Analyzes sequence of matrix updates to infer query properties"""

        # Track matrix changes over time
        for round_info in selection_rounds:
            analysis = self.analyze_matrix_update(round_info)
            self.matrix_changes.append(analysis)

        # Perform temporal analysis
        temporal_patterns = self._analyze_temporal_patterns()

        # Identify important subspace
        query_subspace = self._identify_query_subspace()

        # Cluster analysis in the important subspace
        cluster_analysis = self._perform_cluster_analysis(query_subspace)

        return {
            'temporal_patterns': temporal_patterns,
            'query_subspace': query_subspace,
            'cluster_analysis': cluster_analysis,
            'reconstructed_direction': self._estimate_query_direction()
        }

    def _analyze_temporal_patterns(self) -> Dict:
        """Analyzes how the information matrix changes over time"""

        # Stack principal directions
        directions = np.stack([c['principal_direction'] for c in self.matrix_changes])

        # Compute consistency of directions
        direction_consistency = np.abs(directions @ directions.T)

        # Analyze convergence
        eigenvalue_sequence = np.array([c['eigenvalues'][0] for c in self.matrix_changes])
        convergence_rate = np.diff(eigenvalue_sequence)

        return {
            'direction_consistency': direction_consistency,
            'convergence_rate': convergence_rate
        }

    def _identify_query_subspace(self) -> np.ndarray:
        """Identifies important subspace related to query"""

        # Weight principal directions by their eigenvalues
        weighted_directions = np.stack([
            c['principal_direction'] * np.abs(c['eigenvalues'][0])
            for c in self.matrix_changes
        ])

        # Perform PCA to find dominant subspace
        pca = PCA(n_components=min(3, len(weighted_directions)))
        pca.fit(weighted_directions)

        return pca.components_

    def _perform_cluster_analysis(self, query_subspace: np.ndarray) -> Dict:
        """Analyzes clustering structure in query subspace"""

        # Project seller points to query subspace
        projected_points = self.seller_embeddings @ query_subspace.T

        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(projected_points)

        # Analyze cluster properties
        cluster_stats = {
            'centers': kmeans.cluster_centers_,
            'sizes': np.bincount(clusters),
            'influence_by_cluster': [
                np.mean(self.influence_scores[clusters == i])
                for i in range(self.n_clusters)
            ]
        }

        # Find most influential cluster
        target_cluster = np.argmax(cluster_stats['influence_by_cluster'])

        return {
            'clusters': clusters,
            'stats': cluster_stats,
            'target_cluster': target_cluster
        }

    def _estimate_query_direction(self) -> np.ndarray:
        """Estimates direction of query in embedding space"""

        # Weight directions by eigenvalues and consistency
        patterns = self._analyze_temporal_patterns()
        weights = np.array([
            np.abs(c['eigenvalues'][0]) * np.mean(patterns['direction_consistency'][i])
            for i, c in enumerate(self.matrix_changes)
        ])

        # Compute weighted average of principal directions
        directions = np.stack([c['principal_direction'] for c in self.matrix_changes])
        query_direction = np.average(directions, weights=weights, axis=0)

        # Normalize
        return query_direction / np.linalg.norm(query_direction)

    def reconstruct_query_properties(self) -> Dict:
        """Attempts to reconstruct properties of the query"""

        # Get most influential points
        top_k = 10
        top_indices = np.argsort(self.influence_scores)[-top_k:]

        # Estimate query location
        estimated_center = np.mean(self.seller_embeddings[top_indices], axis=0)

        # Estimate query direction
        query_direction = self._estimate_query_direction()

        # Project all points onto query direction
        projections = self.seller_embeddings @ query_direction

        # Find range of likely query values
        projection_range = (np.percentile(projections, 25),
                            np.percentile(projections, 75))

        return {
            'estimated_center': estimated_center,
            'query_direction': query_direction,
            'projection_range': projection_range,
            'top_similar_points': top_indices,
            'influence_scores': self.influence_scores
        }


# Helper class for evaluating attack success
class InfoMatrixAttackEvaluator:
    def __init__(self, true_query: np.ndarray):
        self.true_query = true_query

    def evaluate_reconstruction(self,
                                reconstruction: Dict,
                                seller_embeddings: np.ndarray) -> Dict[str, float]:
        """Evaluates success of the attack"""

        # Compute distance to true query
        estimated_center = reconstruction['estimated_center']
        center_distance = np.linalg.norm(estimated_center - self.true_query)

        # Compute direction alignment
        direction_alignment = np.abs(
            reconstruction['query_direction'] @
            (self.true_query / np.linalg.norm(self.true_query))
        )

        # Check if true query is in the projected range
        projection = self.true_query @ reconstruction['query_direction']
        in_range = (reconstruction['projection_range'][0] <= projection <=
                    reconstruction['projection_range'][1])

        # Compute nearest neighbor rank
        distances = np.linalg.norm(seller_embeddings - self.true_query, axis=1)
        true_nearest = np.argsort(distances)[:10]
        recovered_nearest = reconstruction['top_similar_points']
        overlap = len(set(true_nearest) & set(recovered_nearest))

        return {
            'center_distance': center_distance,
            'direction_alignment': direction_alignment,
            'in_projection_range': in_range,
            'nearest_neighbor_overlap': overlap / 10,
            'overall_success': (direction_alignment + (overlap / 10)) / 2
        }
