import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.attacks import InfoMatrixAttack


class AttackType(Enum):
    INFO_MATRIX = "info_matrix"
    GRADIENT = "gradient"
    SELECTION_PATTERN = "selection_pattern"
    ENSEMBLE = "ensemble"


@dataclass
class DAVEDOutput:
    """Standardized output from DAVED selection"""
    selected_indices: np.ndarray
    selection_weights: np.ndarray
    gradients: Optional[np.ndarray]
    info_matrix: Optional[np.ndarray]
    round_number: int


@dataclass
class ExperimentConfig:
    """Configuration for attack experiments"""
    n_queries: int = 100  # Number of queries to test
    n_rounds: int = 10  # Number of selection rounds
    query_types: List[str] = None  # Types of query sampling
    attack_types: List[AttackType] = None
    embedding_dims: List[int] = None
    n_seller_points: List[int] = None
    noise_levels: List[float] = None


class AttackPipeline:
    """Main pipeline for running attacks against DAVED"""

    def __init__(self,
                 daved_func: Callable,
                 config: ExperimentConfig):
        self.daved_func = daved_func
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize attacks
        self.attacks = {
            AttackType.INFO_MATRIX: InfoMatrixAttack(),
            AttackType.GRADIENT: GradientBasedAttack(),
            AttackType.SELECTION_PATTERN: SelectionPatternAttack(),
            AttackType.ENSEMBLE: EnsembleAttack()
        }

    def generate_experiment_data(self,
                                 n_points: int,
                                 dim: int,
                                 query_type: str,
                                 noise_level: float = 0.0) -> Dict:
        """Generates data for one experiment"""

        # Generate seller embeddings
        seller_embeddings = np.random.randn(n_points, dim)
        seller_embeddings = StandardScaler().fit_transform(seller_embeddings)

        # Generate query based on type
        if query_type == "random":
            query = np.random.randn(dim)
            query = query / np.linalg.norm(query)

        elif query_type == "cluster":
            # Sample from a specific cluster
            center = np.random.randn(dim)
            query = center + np.random.randn(dim) * 0.1
            query = query / np.linalg.norm(query)

        elif query_type == "mixture":
            # Sample from mixture of distributions
            components = np.random.randn(3, dim)
            weights = np.random.dirichlet(np.ones(3))
            query = weights @ components
            query = query / np.linalg.norm(query)

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.randn(dim) * noise_level
            query = query + noise
            query = query / np.linalg.norm(query)

        return {
            'seller_embeddings': seller_embeddings,
            'query': query
        }

    def run_single_experiment(self,
                              seller_embeddings: np.ndarray,
                              query: np.ndarray,
                              attack_type: AttackType) -> Dict:
        """Runs single experiment with one attack type"""

        # Run DAVED selection
        daved_outputs = []
        for round_num in range(self.config.n_rounds):
            output = self.daved_func(
                seller_embeddings=seller_embeddings,
                query=query,
                round_number=round_num
            )
            daved_outputs.append(output)

        # Run attack
        attack = self.attacks[attack_type]
        attack_result = attack.run_attack(daved_outputs, seller_embeddings)

        # Evaluate attack success
        evaluator = AttackEvaluator(true_query=query)
        metrics = evaluator.evaluate(attack_result, seller_embeddings)

        return {
            'attack_result': attack_result,
            'metrics': metrics
        }

    def run_experiments(self) -> Dict:
        """Runs full set of experiments"""
        results = {}

        for dim in self.config.embedding_dims:
            for n_points in self.config.n_seller_points:
                for query_type in self.config.query_types:
                    for noise_level in self.config.noise_levels:
                        # Generate experiment key
                        exp_key = f"dim{dim}_points{n_points}_{query_type}_noise{noise_level}"
                        results[exp_key] = {attack: [] for attack in self.config.attack_types}

                        self.logger.info(f"Running experiment: {exp_key}")

                        # Run multiple queries
                        for query_idx in range(self.config.n_queries):
                            # Generate data
                            data = self.generate_experiment_data(
                                n_points=n_points,
                                dim=dim,
                                query_type=query_type,
                                noise_level=noise_level
                            )

                            # Run each attack type
                            for attack_type in self.config.attack_types:
                                exp_result = self.run_single_experiment(
                                    seller_embeddings=data['seller_embeddings'],
                                    query=data['query'],
                                    attack_type=attack_type
                                )
                                results[exp_key][attack_type].append(exp_result)

        return results

    def analyze_results(self, results: Dict) -> Dict:
        """Analyzes experimental results"""
        analysis = {}

        for exp_key, exp_results in results.items():
            analysis[exp_key] = {}

            for attack_type, attack_results in exp_results.items():
                metrics = [r['metrics'] for r in attack_results]

                # Compute statistics
                analysis[exp_key][attack_type] = {
                    'mean_success': np.mean([m['success_score'] for m in metrics]),
                    'std_success': np.std([m['success_score'] for m in metrics]),
                    'mean_distance': np.mean([m['distance'] for m in metrics]),
                    'privacy_breach_rate': np.mean([m['privacy_breached'] for m in metrics])
                }

        return analysis

    def visualize_results(self, results: Dict, save_path: str = None):
        """Visualizes attack results"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create visualization_226 for each metric
        metrics = ['mean_success', 'privacy_breach_rate']

        for metric in metrics:
            plt.figure(figsize=(12, 6))
            data = []

            for exp_key, exp_results in results.items():
                for attack_type, stats in exp_results.items():
                    data.append({
                        'experiment': exp_key,
                        'attack': attack_type.value,
                        metric: stats[metric]
                    })

            df = pd.DataFrame(data)
            sns.barplot(x='experiment', y=metric, hue='attack', data=df)
            plt.xticks(rotation=45)
            plt.title(f'{metric} across experiments and attacks')

            if save_path:
                plt.savefig(f"{save_path}/{metric}.png")
            plt.close()


class AttackEvaluator:
    """Evaluates attack success metrics"""

    def __init__(self, true_query: np.ndarray):
        self.true_query = true_query

    def evaluate(self,
                 attack_result: Dict,
                 seller_embeddings: np.ndarray) -> Dict:
        """Evaluates attack success"""

        # Calculate basic metrics
        predicted_query = attack_result['predicted_query']
        distance = np.linalg.norm(predicted_query - self.true_query)
        cosine_sim = np.dot(predicted_query, self.true_query) / (
                np.linalg.norm(predicted_query) * np.linalg.norm(self.true_query)
        )

        # Check if privacy is breached (cosine similarity above threshold)
        privacy_breached = cosine_sim > 0.8

        # Calculate neighborhood preservation
        true_nn = self._get_nearest_neighbors(seller_embeddings, self.true_query)
        pred_nn = self._get_nearest_neighbors(seller_embeddings, predicted_query)
        nn_overlap = len(set(true_nn) & set(pred_nn)) / len(true_nn)

        # Overall success score
        success_score = (cosine_sim + nn_overlap) / 2

        return {
            'distance': distance,
            'cosine_similarity': cosine_sim,
            'privacy_breached': privacy_breached,
            'nn_overlap': nn_overlap,
            'success_score': success_score
        }

    def _get_nearest_neighbors(self,
                               points: np.ndarray,
                               query: np.ndarray,
                               k: int = 10) -> np.ndarray:
        """Gets indices of k nearest neighbors"""
        distances = np.linalg.norm(points - query, axis=1)
        return np.argsort(distances)[:k]
