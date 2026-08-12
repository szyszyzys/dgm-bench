import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

@dataclass
class SellerDataStructure:
    """Strategic structure of seller's data"""
    embeddings: np.ndarray  # Original embeddings
    cluster_labels: np.ndarray  # Cluster assignments
    landmark_points: np.ndarray  # Strategically placed landmarks
    local_structures: Dict  # Local neighborhood structures
    coverage_map: np.ndarray  # Space coverage analysis

@dataclass
class SelectionInfo:
    """Information from one selection round"""
    selected_indices: np.ndarray
    round_number: int
    weights: Optional[np.ndarray] = None

class StrategicDataOrganizer:
    """Organizes seller data strategically for better attack"""

    def __init__(self,
                 embeddings: np.ndarray,
                 n_landmarks: int = 100,
                 n_clusters: int = 20,
                 neighbor_k: int = 10):
        self.embeddings = embeddings
        self.n_landmarks = n_landmarks
        self.n_clusters = n_clusters
        self.neighbor_k = neighbor_k

    def organize_data(self) -> SellerDataStructure:
        """Strategically organize seller's data"""
        # 1. Create landmark points
        landmarks = self._create_landmarks()

        # 2. Cluster the space
        cluster_labels = self._cluster_space()

        # 3. Analyze local structures
        local_structures = self._analyze_local_structures()

        # 4. Create coverage map
        coverage_map = self._create_coverage_map()

        return SellerDataStructure(
            embeddings=self.embeddings,
            cluster_labels=cluster_labels,
            landmark_points=landmarks,
            local_structures=local_structures,
            coverage_map=coverage_map
        )

    def _create_landmarks(self) -> np.ndarray:
        """Create strategic landmark points"""
        # Use K-means++ initialization to place landmarks
        kmeans = KMeans(n_clusters=self.n_landmarks, init='k-means++')
        kmeans.fit(self.embeddings)
        landmarks = kmeans.cluster_centers_

        # Add boundary points
        pca = PCA(n_components=2)
        projected = pca.fit_transform(self.embeddings)
        for dim in range(2):
            extremes = [np.argmin(projected[:, dim]), np.argmax(projected[:, dim])]
            landmarks = np.vstack([landmarks, self.embeddings[extremes]])

        return landmarks

    def _cluster_space(self) -> np.ndarray:
        """Cluster the embedding space"""
        kmeans = KMeans(n_clusters=self.n_clusters)
        return kmeans.fit_predict(self.embeddings)

    def _analyze_local_structures(self) -> Dict:
        """Analyze local neighborhood structures"""
        nn = NearestNeighbors(n_neighbors=self.neighbor_k)
        nn.fit(self.embeddings)

        local_structures = {}
        distances, indices = nn.kneighbors(self.embeddings)

        for i in range(len(self.embeddings)):
            # Compute local statistics
            local_points = self.embeddings[indices[i]]
            center = np.mean(local_points, axis=0)
            spread = np.std(local_points, axis=0)

            # Compute local PCA
            local_pca = PCA(n_components=min(3, self.embeddings.shape[1]))
            local_pca.fit(local_points)

            local_structures[i] = {
                'neighbors': indices[i],
                'distances': distances[i],
                'center': center,
                'spread': spread,
                'principal_directions': local_pca.components_,
                'explained_variance': local_pca.explained_variance_ratio_
            }

        return local_structures

    def _create_coverage_map(self) -> np.ndarray:
        """Create space coverage analysis"""
        # Project to 2D for visualization_226
        tsne = TSNE(n_components=2)
        projected = tsne.fit_transform(self.embeddings)

        # Create density map
        x_grid = np.linspace(projected[:, 0].min(), projected[:, 0].max(), 50)
        y_grid = np.linspace(projected[:, 1].min(), projected[:, 1].max(), 50)
        coverage_map = np.zeros((len(x_grid), len(y_grid)))

        # Compute density
        for i in range(len(x_grid)-1):
            for j in range(len(y_grid)-1):
                mask = ((projected[:, 0] >= x_grid[i]) &
                       (projected[:, 0] < x_grid[i+1]) &
                       (projected[:, 1] >= y_grid[j]) &
                       (projected[:, 1] < y_grid[j+1]))
                coverage_map[i, j] = np.sum(mask)

        return coverage_map

class SelectionPatternAttack:
    """Advanced attack based on selection patterns"""

    def __init__(self,
                 seller_data: SellerDataStructure,
                 temporal_window: int = 3,
                 min_confidence: float = 0.7):
        self.seller_data = seller_data
        self.temporal_window = temporal_window
        self.min_confidence = min_confidence

        # Initialize tracking
        self.selection_history = []
        self.cluster_history = []
        self.confidence_scores = []

    def analyze_selection(self, selection: SelectionInfo) -> Dict:
        """Analyze one selection round"""
        # Store selection
        self.selection_history.append(selection)

        # Get cluster assignments for selected points
        selected_clusters = self.seller_data.cluster_labels[selection.selected_indices]
        self.cluster_history.append(selected_clusters)

        # Analyze local structures
        local_analysis = self._analyze_local_structures(selection)

        # Analyze temporal patterns if enough history
        temporal_analysis = self._analyze_temporal_patterns() if len(self.selection_history) >= self.temporal_window else None

        return {
            'local_analysis': local_analysis,
            'temporal_analysis': temporal_analysis
        }

    def _analyze_local_structures(self, selection: SelectionInfo) -> Dict:
        """Analyze local structures of selected points"""
        local_structures = []

        for idx in selection.selected_indices:
            local_info = self.seller_data.local_structures[idx]

            # Compute selection density in neighborhood
            neighbor_selections = np.isin(
                local_info['neighbors'],
                selection.selected_indices
            )
            selection_density = np.mean(neighbor_selections)

            local_structures.append({
                'center': local_info['center'],
                'spread': local_info['spread'],
                'selection_density': selection_density,
                'principal_directions': local_info['principal_directions']
            })

        return {
            'structures': local_structures,
            'common_directions': self._find_common_directions(local_structures)
        }

    def _analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in selections"""
        window = self.selection_history[-self.temporal_window:]
        clusters = self.cluster_history[-self.temporal_window:]

        # Analyze cluster transitions
        cluster_transitions = []
        for i in range(len(clusters)-1):
            curr_clusters = set(clusters[i])
            next_clusters = set(clusters[i+1])
            transition = {
                'overlap': len(curr_clusters & next_clusters),
                'new_clusters': len(next_clusters - curr_clusters)
            }
            cluster_transitions.append(transition)

        # Analyze selection consistency
        selection_consistency = []
        for i in range(len(window)-1):
            curr_selected = set(window[i].selected_indices)
            next_selected = set(window[i+1].selected_indices)
            consistency = len(curr_selected & next_selected) / len(curr_selected)
            selection_consistency.append(consistency)

        return {
            'cluster_transitions': cluster_transitions,
            'selection_consistency': np.mean(selection_consistency)
        }

    def _find_common_directions(self, local_structures: List[Dict]) -> np.ndarray:
        """Find common directions across local structures"""
        all_directions = []
        for struct in local_structures:
            all_directions.extend(struct['principal_directions'])

        # Cluster directions to find common ones
        direction_clusters = KMeans(n_clusters=min(3, len(all_directions)))
        direction_clusters.fit(all_directions)

        return direction_clusters.cluster_centers_

    def reconstruct_query(self) -> Tuple[np.ndarray, float]:
        """Reconstruct query based on selection patterns"""
        # 1. Analyze cluster patterns
        cluster_frequencies = np.zeros(self.seller_data.n_clusters)
        for clusters in self.cluster_history:
            unique, counts = np.unique(clusters, return_counts=True)
            cluster_frequencies[unique] += counts

        # 2. Find dominant clusters
        dominant_clusters = np.argsort(cluster_frequencies)[-3:]

        # 3. Analyze local structures in dominant clusters
        relevant_points = []
        relevance_weights = []

        for cluster in dominant_clusters:
            cluster_points = self.seller_data.embeddings[
                self.seller_data.cluster_labels == cluster
            ]
            weights = cluster_frequencies[cluster] * np.ones(len(cluster_points))
            relevant_points.append(cluster_points)
            relevance_weights.append(weights)

        relevant_points = np.vstack(relevant_points)
        relevance_weights = np.concatenate(relevance_weights)

        # 4. Estimate query direction
        pca = PCA(n_components=1)
        pca.fit(relevant_points, sample_weight=relevance_weights)
        query_direction = pca.components_[0]

        # 5. Estimate magnitude using landmark points
        projections = self.seller_data.landmark_points @ query_direction
        magnitude = np.std(projections) * np.sqrt(cluster_frequencies.max())

        # 6. Compute confidence score
        temporal_consistency = np.mean([
            analysis['temporal_analysis']['selection_consistency']
            for analysis in self.selection_history
            if analysis.get('temporal_analysis')
        ])

        confidence = temporal_consistency * pca.explained_variance_ratio_[0]

        return query_direction * magnitude, confidence

class SelectionAttackEvaluator:
    """Evaluates success of selection pattern attack"""

    def __init__(self, true_query: np.ndarray):
        self.true_query = true_query

    def evaluate(self,
                reconstructed_query: np.ndarray,
                confidence: float) -> Dict:
        """Evaluate attack success"""
        # Normalize vectors
        true_norm = self.true_query / np.linalg.norm(self.true_query)
        pred_norm = reconstructed_query / np.linalg.norm(reconstructed_query)

        # Compute alignment
        alignment = np.abs(np.dot(true_norm, pred_norm))

        # Compute magnitude error
        magnitude_error = np.abs(
            np.linalg.norm(self.true_query) -
            np.linalg.norm(reconstructed_query)
        ) / np.linalg.norm(self.true_query)

        return {
            'alignment': alignment,
            'magnitude_error': magnitude_error,
            'confidence': confidence,
            'success_score': alignment * (1 - magnitude_error) * confidence
        }