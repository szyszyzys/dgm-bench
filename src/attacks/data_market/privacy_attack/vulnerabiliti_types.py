class VulnerabilityExperiments:
    def __init__(self):
        self.experiments = {
            'memory_attacks': self._setup_memory_attacks(),
            'correlation_attacks': self._setup_correlation_attacks(),
            'reconstruction_attacks': self._setup_reconstruction_attacks(),
            'membership_inference': self._setup_membership_inference()
        }

    def _setup_memory_attacks(self):
        """Experiments testing memory-based vulnerabilities"""
        return {
            'query_history': {
                'window_sizes': [5, 10, 20],
                'history_length': 100,
                'pattern_types': ['periodic', 'random', 'structured']
            },
            'pattern_memory': {
                'pattern_length': 5,
                'noise_levels': [0.0, 0.1, 0.2],
                'repetition_count': 10
            }
        }

    def _setup_correlation_attacks(self):
        """Experiments testing correlation-based vulnerabilities"""
        return {
            'temporal_correlation': {
                'time_windows': [10, 50, 100],
                'correlation_types': ['linear', 'nonlinear']
            },
            'spatial_correlation': {
                'neighborhood_sizes': [5, 10, 20],
                'distance_metrics': ['euclidean', 'cosine']
            }
        }

    def _setup_reconstruction_attacks(self):
        """Experiments for query reconstruction attacks"""
        return {
            'direct_reconstruction': {
                'methods': ['gradient', 'pattern', 'matrix'],
                'confidence_thresholds': [0.7, 0.8, 0.9]
            },
            'indirect_reconstruction': {
                'feature_types': ['explicit', 'implicit', 'derived'],
                'reconstruction_levels': ['coarse', 'fine']
            }
        }

    def _setup_membership_inference(self):
        """Experiments for membership inference attacks"""
        return {
            'query_membership': {
                'dataset_sizes': [1000, 5000, 10000],
                'query_overlap': [0.0, 0.2, 0.4]
            },
            'attribute_inference': {
                'attribute_types': ['categorical', 'numerical', 'mixed'],
                'inference_methods': ['direct', 'indirect']
            }
        }


class PrivacyMetrics:
    """Privacy leakage quantification metrics"""

    @staticmethod
    def query_reconstruction_error(true_query: np.ndarray,
                                   reconstructed_query: np.ndarray) -> float:
        """Measures query reconstruction accuracy"""
        return np.linalg.norm(true_query - reconstructed_query)

    @staticmethod
    def pattern_detection_rate(true_patterns: List,
                               detected_patterns: List) -> float:
        """Measures pattern detection accuracy"""
        return len(set(true_patterns) & set(detected_patterns)) / len(true_patterns)

    @staticmethod
    def information_gain(prior_entropy: float,
                         posterior_entropy: float) -> float:
        """Measures information gained through attacks"""
        return prior_entropy - posterior_entropy

    @staticmethod
    def privacy_loss(original_privacy: float,
                     current_privacy: float) -> float:
        """Quantifies privacy loss"""
        return original_privacy - current_privacy
