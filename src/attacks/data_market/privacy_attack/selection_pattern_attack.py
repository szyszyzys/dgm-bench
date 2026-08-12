import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter

@dataclass
class SelectionPattern:
    """Information from one round of DAVED selection"""
    selected_indices: np.ndarray  # Indices of selected points
    selection_weights: np.ndarray  # Selection weights
    round_number: int  # Round number
    distances: np.ndarray  # Distances between points (optional)

class SelectionPatternAttack:
    """
    Attack that exploits patterns in point selection over time
    """
    def __init__(self,
                 seller_embeddings: np.ndarray,
                 n_clusters: int = 5,
                 history_window: int = 3,
                 neighbor_k: int = 10):
        self.seller_embeddings = seller_embeddings
        self.n_clusters = n_clusters
        self.history_window = history_window
        self.neighbor_k = neighbor_k

        # Initialize tracking structures
        self.selection_history = []
        self.weight_history = []
        self.frequency_map = np.zeros(len(seller_embeddings))

        # Initialize neighbor finder
        self.nn = NearestNeighbors(n_neighbors=neighbor_k)
        self.nn.fit(seller_embeddings)

        # Track selection patterns
        self.sequential_patterns = []
        self.cluster_assignments = None

    def analyze_selection_round(self, round_info: SelectionPattern) -> Dict:
        """Analyzes a single round of selection"""
        # Update histories
        self.selection_history.append(round_info.selected_indices)
        self.weight_history.append(round_info.selection_weights)

        # Update frequency map
        self.frequency_map[round_info.selected_indices] += round_info.selection_weights

        # Analyze current round patterns
        current_patterns = self._analyze_current_selection(round_info)

        # Analyze sequential patterns if we have enough history
        if len(self.selection_history) >= self.history_window:
            sequential_patterns = self._analyze_sequential_patterns()
        else:
            sequential_patterns = None

        # Analyze spatial patterns
        spatial_patterns = self._analyze_spatial_patterns(round_info)

        return {
            'current_patterns': current_patterns,
            'sequential_patterns': sequential_patterns,
            'spatial_patterns': spatial_patterns,
            'round_number': round_info.round_number
        }

    def _analyze_current_selection(self, round_info: SelectionPattern) -> Dict:
        """Analyzes patterns in current selection round"""
        selected_points = self.seller_embeddings[round_info.selected_indices]

        # Compute centroid and spread of selected points
        centroid = np.average(selected_points, weights=round_info.selection_weights, axis=0)
        distances_to_centroid = np.linalg.norm(selected_points - centroid, axis=1)

        # Analyze local neighborhood of selected points
        neighborhood_stats = []
        for idx in round_info.selected_indices:
            distances, neighbors = self.nn.kneighbors([self.seller_embeddings[idx]])
            selection_ratio = np.mean(np.isin(neighbors[0], round_info.selected_indices))
            neighborhood_stats.append({
                'point_idx': idx,
                'neighbors': neighbors[0],
                'selection_ratio': selection_ratio
            })

        return {
            'centroid': centroid,
            'spread': np.mean(distances_to_centroid),
            'max_weight': np.max(round_info.selection_weights),
            'neighborhood_stats': neighborhood_stats
        }

    def _analyze_sequential_patterns(self) -> Dict:
        """Analyzes patterns in selection sequence"""
        window = self.history_window
        recent_selections = self.selection_history[-window:]
        recent_weights = self.weight_history[-window:]

        # Analyze selection transitions
        transitions = []
        for i in range(len(recent_selections)-1):
            curr_selected = set(recent_selections[i])
            next_selected = set(recent_selections[i+1])
            overlap = len(curr_selected & next_selected)
            transitions.append({
                'overlap': overlap,
                'ratio': overlap / len(curr_selected)
            })

        # Find consistent selections
        consistent_points = set.intersection(*map(set, recent_selections))

        # Analyze weight evolution
        weight_evolution = {}
        for idx in set.union(*map(set, recent_selections)):
            weights = []
            for i, selection in enumerate(recent_selections):
                if idx in selection:
                    weight_idx = np.where(selection == idx)[0][0]
                    weights.append(recent_weights[i][weight_idx])
                else:
                    weights.append(0)
            weight_evolution[idx] = weights

        return {
            'transitions': transitions,
            'consistent_points': list(consistent_points),
            'weight_evolution': weight_evolution
        }

    def _analyze_spatial_patterns(self, round_info: SelectionPattern) -> Dict:
        """Analyzes spatial patterns in selections"""
        # Cluster all points
        if self.cluster_assignments is None:
            kmeans = KMeans(n_clusters=self.n_clusters)
            self.cluster_assignments = kmeans.fit_predict(self.seller_embeddings)
            self.cluster_centers = kmeans.cluster_centers_

        # Analyze cluster distribution of selected points
        selected_clusters = self.cluster_assignments[round_info.selected_indices]
        cluster_counts = Counter(selected_clusters)

        # Find dominant clusters
        cluster_weights = np.zeros(self.n_clusters)
        for cluster_id, count in cluster_counts.items():
            cluster_weights[cluster_id] = count

        return {
            'cluster_distribution': dict(cluster_counts),
            'cluster_weights': cluster_weights
        }

    def estimate_query_region(self) -> Dict:
        """Estimates region where query might lie based on selection patterns"""
        # Weight points by selection frequency
        normalized_freq = self.frequency_map / np.sum(self.frequency_map)

        # Get high-frequency points
        top_k = min(self.neighbor_k, len(self.seller_embeddings))
        top_indices = np.argsort(normalized_freq)[-top_k:]
        top_points = self.seller_embeddings[top_indices]

        # Estimate center and radius of query region
        query_center = np.average(top_points, weights=normalized_freq[top_indices], axis=0)

        # Compute radius as weighted standard deviation
        distances = np.linalg.norm(top_points - query_center, axis=1)
        query_radius = np.average(distances, weights=normalized_freq[top_indices])

        # Find principal directions of variation
        pca = PCA(n_components=min(3, len(top_points)))
        pca.fit(top_points)

        # Analyze cluster presence
        if self.cluster_assignments is not None:
            top_clusters = Counter(self.cluster_assignments[top_indices])
        else:
            top_clusters = None

        return {
            'query_center': query_center,
            'query_radius': query_radius,
            'principal_directions': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
            'high_frequency_points': top_indices,
            'dominant_clusters': top_clusters
        }

    def predict_next_selection(self) -> np.ndarray:
        """Predicts which points are likely to be selected next"""
        if len(self.selection_history) < 2:
            return None

        # Analyze recent selection patterns
        recent_selections = set(self.selection_history[-1])

        # Find points similar to recently selected ones
        likely_points = set()
        for idx in recent_selections:
            _, neighbors = self.nn.kneighbors([self.seller_embeddings[idx]])
            likely_points.update(neighbors[0])

        # Score points by frequency and recency
        scores = np.zeros(len(self.seller_embeddings))
        for i in likely_points:
            # Frequency score
            scores[i] += self.frequency_map[i]

            # Recency score
            for t, selection in enumerate(self.selection_history[-self.history_window:]):
                if i in selection:
                    scores[i] += (t + 1) / self.history_window

        return scores / np.max(scores)

class SelectionPatternEvaluator:
    """Evaluates success of selection pattern attack"""

    def __init__(self, true_query: np.ndarray):
        self.true_query = true_query

    def evaluate_region_estimate(self, region_info: Dict) -> Dict[str, float]:
        """Evaluates quality of region estimation"""
        # Compute distance to estimated center
        center_distance = np.linalg.norm(self.true_query - region_info['query_center'])

        # Check if query is within estimated region
        in_region = center_distance <= region_info['query_radius']

        # Compute alignment with principal directions
        direction_alignment = np.mean([
            abs(np.dot(self.true_query, direction))
            for direction in region_info['principal_directions']
        ])

        # Compute overall success metric
        success_score = (
            0.4 * (1.0 if in_region else 0.0) +
            0.3 * (1.0 - center_distance / np.linalg.norm(self.true_query)) +
            0.3 * direction_alignment
        )

        return {
            'center_distance': center_distance,
            'in_region': in_region,
            'direction_alignment': direction_alignment,
            'success_score': success_score
        }

    def evaluate_prediction(self,
                          predicted_scores: np.ndarray,
                          actual_selection: np.ndarray) -> Dict[str, float]:
        """Evaluates quality of next-selection prediction"""
        if predicted_scores is None:
            return None

        # Compute prediction accuracy
        top_k = len(actual_selection)
        predicted_indices = np.argsort(predicted_scores)[-top_k:]
        overlap = len(set(predicted_indices) & set(actual_selection))

        return {
            'prediction_accuracy': overlap / top_k,
            'top_k_precision': overlap / len(predicted_indices)
        }