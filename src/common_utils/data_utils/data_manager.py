import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List, Tuple

# TODO: daved module was removed — re-enable when a replacement is available
# from src.mechanism.data.discovery.daved.src import get_data_data_market


class DatasetManager:
    """Manages dataset loading, preprocessing, and distribution among sellers"""

    def __init__(self,
                 dataset_type: str = "gaussian",
                 data_dir: str = "./data",
                 random_state: int = 0,
                 num_seller: int = 10000,
                 num_buyer: int = 100,
                 num_val: int = 100,
                 dim: int = 100,
                 noise_level: float = 1.0,
                 cost_range: Optional[List[float]] = None,
                 cost_func: str = "linear",
                 buyer_query_selection_method: str = "random",  # New parameter
                 selection_params: Optional[Dict] = None,  # Parameters for selection methods
                 use_cost=False
                 ):

        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.seller_data = {}

        # Load data using get_data function
        self.data = get_data_data_market(
            dataset=dataset_type,
            data_dir=data_dir,
            random_state=random_state,
            num_seller=num_seller,
            num_buyer=num_buyer,
            num_val=num_val,
            dim=dim,
            noise_level=noise_level,
            cost_range=cost_range,
            cost_func=cost_func
        )

        # Extract components
        self.X_sell = self.data["X_sell"]
        self.y_sell = self.data["y_sell"]
        self.costs_sell = self.data.get("costs_sell") if use_cost else None
        self.index_sell = self.data.get("index_sell")
        self.img_paths = self.data.get("img_paths")

        # Buyer data
        self.X_buy_full = self.data["X_buy"]  # Full buyer pool before selection
        self.y_buy_full = self.data["y_buy"]
        self.index_buy_full = self.data.get("index_buy")

        # Validation data
        self.X_val = self.data.get("X_val")
        self.y_val = self.data.get("y_val")
        self.index_val = self.data.get("index_val")

        # Set selection method
        self.buyer_query_selection_method = buyer_query_selection_method
        self.buyer_selection_params = selection_params if selection_params else {}

        # Perform buyer data selection
        self.X_buy, self.y_buy, self.index_buy = self.select_buyer_data()

    def allocate_data_to_sellers(self,
                                 seller_configs: List[Dict],
                                 adversary_ratio: float = 0.1,
                                 overlap_allowed: bool = False) -> Dict[str, Dict]:
        """
        Allocate data to sellers based on configurations
        
        Parameters:
        - seller_configs: List of seller configurations with 'id' and 'type' ('normal' or 'adversary')
        - adversary_ratio: Ratio of data to allocate to adversarial sellers
        - overlap_allowed: Whether sellers can have overlapping data
        
        Returns:
        - Dictionary mapping seller IDs to their allocated data
        """
        np.random.seed(self.random_state)
        n_samples = len(self.X_sell)

        # Separate adversarial and normal sellers
        adversary_sellers = [s for s in seller_configs if s['type'] == 'adversary']
        normal_sellers = [s for s in seller_configs if s['type'] == 'normal']

        # Calculate sizes
        adv_size = int(n_samples * adversary_ratio)
        normal_size = n_samples - adv_size

        # Create index pools
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)

        adv_indices = all_indices[:adv_size]
        normal_indices = all_indices[adv_size:]

        allocations = {}

        # Allocate to adversaries
        if adversary_sellers:
            indices_per_adv = adv_size // len(adversary_sellers)
            for i, seller in enumerate(adversary_sellers):
                start_idx = i * indices_per_adv
                end_idx = start_idx + indices_per_adv
                seller_indices = adv_indices[start_idx:end_idx]

                allocations[seller['id']] = {
                    'X': self.X_sell[seller_indices],
                    'y': self.y_sell[seller_indices],
                    'costs': self.costs_sell[seller_indices] if self.costs_sell is not None else None,
                    'indices': self.index_sell[seller_indices] if self.index_sell is not None else None,
                }

        # Allocate to normal sellers
        if normal_sellers:
            indices_per_normal = normal_size // len(normal_sellers)
            for i, seller in enumerate(normal_sellers):
                if overlap_allowed:
                    # Random selection with possible overlap
                    seller_indices = np.random.choice(normal_indices, size=indices_per_normal)
                else:
                    # Sequential allocation without overlap
                    start_idx = i * indices_per_normal
                    end_idx = start_idx + indices_per_normal
                    seller_indices = normal_indices[start_idx:end_idx]

                allocations[seller['id']] = {
                    'X': self.X_sell[seller_indices],
                    'y': self.y_sell[seller_indices],
                    'costs': self.costs_sell[seller_indices] if self.costs_sell is not None else None,
                    'indices': self.index_sell[seller_indices] if self.index_sell is not None else None,
                    # 'img_paths': [self.img_paths[i] for i in seller_indices] if self.img_paths is not None else None
                }

        self.seller_data = allocations
        return allocations

    def get_buyer_data(self) -> Dict:
        """Get buyer data"""
        return {
            'X': self.X_buy,
            'y': self.y_buy,
            'indices': self.index_buy
        }

    def get_validation_data(self) -> Dict:
        """Get validation data"""
        return {
            'X': self.X_val,
            'y': self.y_val,
            'indices': self.index_val
        }

    def get_seller_data(self, seller_id: str) -> Optional[Dict]:
        """Get data for specific seller"""
        return self.seller_data.get(seller_id)

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_sellers': len(self.seller_data),
            'num_buyer_samples': len(self.X_buy),
            'num_val_samples': len(self.X_val) if self.X_val is not None else 0,
            'feature_dim': self.X_sell.shape[1],
            'has_costs': self.costs_sell is not None,
            'has_images': self.img_paths is not None
        }

    def select_buyer_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select buyer data based on the specified selection method."""
        if self.buyer_query_selection_method == "random":
            return self._random_selection()
        elif self.buyer_query_selection_method == "cluster":
            return self._cluster_based_selection()
        elif self.buyer_query_selection_method == "diversity":
            return self._diversity_based_selection()
        elif self.buyer_query_selection_method == "uncertainty":
            return self._uncertainty_based_selection()
        elif self.buyer_query_selection_method == "stratified":
            return self._stratified_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.buyer_query_selection_method}")

    def _random_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly select buyer data."""
        num_buyer = self.X_buy_full.shape[0]
        np.random.seed(self.random_state)
        indices = np.random.choice(num_buyer, size=num_buyer, replace=False)
        return (self.X_buy_full[indices],
                self.y_buy_full[indices],
                self.index_buy_full[indices])

    def _cluster_based_selection(self, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select buyer data using K-Means clustering to ensure coverage of all clusters."""
        if n_clusters is None:
            n_clusters = self.buyer_selection_params.get("n_clusters", 10)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_buy_full)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        kmeans.fit(X_scaled)
        cluster_centers = kmeans.cluster_centers_

        # Find the nearest data point to each cluster center
        closest, _ = pairwise_distances_argmin_min(cluster_centers, X_scaled)
        selected_indices = closest

        # If more buyers are needed, fill the rest randomly
        num_selected = len(selected_indices)
        total_buyers = self.X_buy_full.shape[0]
        if num_selected < total_buyers:
            remaining = total_buyers - num_selected
            additional_indices = np.random.choice(
                [i for i in range(total_buyers) if i not in selected_indices],
                size=remaining,
                replace=False
            )
            selected_indices = np.concatenate([selected_indices, additional_indices])

        return (self.X_buy_full[selected_indices],
                self.y_buy_full[selected_indices],
                self.index_buy_full[selected_indices])

    def _diversity_based_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select buyer data to maximize diversity.
        Here, we use farthest point sampling as an example.
        """
        from sklearn.metrics import pairwise_distances

        num_buyer = self.X_buy_full.shape[0]
        X = self.X_buy_full
        selected_indices = []
        remaining_indices = list(range(X.shape[0]))

        # Initialize with a random point
        np.random.seed(self.random_state)
        first_index = np.random.choice(remaining_indices)
        selected_indices.append(first_index)
        remaining_indices.remove(first_index)

        for _ in range(1, num_buyer):
            last_selected = X[selected_indices[-1]].reshape(1, -1)
            distances = pairwise_distances(X[remaining_indices], last_selected).reshape(-1)
            farthest_index = np.argmax(distances)
            selected = remaining_indices[farthest_index]
            selected_indices.append(selected)
            remaining_indices.remove(selected)

        return (self.X_buy_full[selected_indices],
                self.y_buy_full[selected_indices],
                self.index_buy_full[selected_indices])

    def _uncertainty_based_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select buyer data based on model uncertainty.
        This requires a pre-trained model to estimate uncertainty.
        Here, we provide a placeholder implementation.
        """
        # Placeholder: Select based on uncertainty scores
        # Replace this with your actual uncertainty estimation logic
        num_buyer = self.X_buy_full.shape[0]
        # Simulate uncertainty scores
        np.random.seed(self.random_state)
        uncertainty_scores = np.random.rand(num_buyer)
        # Select top uncertain samples
        sorted_indices = np.argsort(-uncertainty_scores)
        return (self.X_buy_full[sorted_indices],
                self.y_buy_full[sorted_indices],
                self.index_buy_full[sorted_indices])

    def _stratified_selection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select buyer data maintaining the same class distribution as the full buyer pool."""
        from sklearn.model_selection import StratifiedShuffleSplit

        num_buyer = self.X_buy_full.shape[0]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=num_buyer, random_state=self.random_state)
        for _, selected_indices in splitter.split(self.X_buy_full, self.y_buy_full):
            selected_indices = selected_indices
            break
        return (self.X_buy_full[selected_indices],
                self.y_buy_full[selected_indices],
                self.index_buy_full[selected_indices])
