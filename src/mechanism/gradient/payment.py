from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.common_utils import cosine_similarity


# from common.utils import cosine_similarity

class PaymentSimulator:
    """
    A modular class to simulate different marketplace payment models.

    This class takes logged experiment data and adds a 'payment_received' column
    to each seller's history DataFrame based on a specified payment scheme.
    This allows for post-hoc analysis of payment-based privacy leakages.
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initializes the PaymentSimulator with a specific payment model.

        Args:
            model_name (str): The name of the payment model to use.
                              Supported: 'binary', 'proportional', 'quality'.
            **kwargs: Model-specific parameters.
                - For 'binary': base_payment (float), noise_level (float)
                - For 'proportional': scale_factor (float), noise_level (float)
                - For 'quality': base_rate (float), similarity_bonus (float)
        """
        self.model_name = model_name
        self.params = kwargs
        self.payment_function = self._get_payment_function()

    def _get_payment_function(self):
        """Maps the model name to the corresponding simulation method."""
        model_map = {
            'binary': self._simulate_binary_payment,
            'proportional': self._simulate_proportional_payment,
            'quality': self._calculate_quality_based_payment,
        }
        if self.model_name not in model_map:
            raise ValueError(f"Unknown payment model: '{self.model_name}'. "
                             f"Supported models are: {list(model_map.keys())}")
        return model_map[self.model_name]

    def add_payments_to_history(
            self,
            seller_history: pd.DataFrame,
            gradients: Optional[Dict[int, np.ndarray]] = None,
            reference_gradients: Optional[Dict[int, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Augments a seller's history DataFrame with a 'payment_received' column.

        Args:
            seller_history (pd.DataFrame): The seller's round-by-round history.
            gradients (Optional[Dict[int, np.ndarray]]): A dictionary mapping
                round number to the seller's flattened gradient for that round.
                Required for the 'quality' model.
            reference_gradients (Optional[Dict[int, np.ndarray]]): A dictionary mapping
                round number to the reference (e.g., aggregated) gradient.
                Required for the 'quality' model.

        Returns:
            pd.DataFrame: The augmented DataFrame with a 'payment_received' column.
        """
        # For simple models that don't need gradients
        if self.model_name in ['binary', 'proportional']:
            return self.payment_function(seller_history)

        # For the quality model that requires gradients
        elif self.model_name == 'quality':
            if gradients is None or reference_gradients is None:
                raise ValueError("'quality' model requires gradients and reference_gradients.")

            payments = []
            for index, row in seller_history.iterrows():
                round_num = int(row['round'])
                payment = 0.0
                if row['assigned_weight'] > 0 and round_num in gradients:
                    grad = gradients.get(round_num)
                    ref_grad = reference_gradients.get(round_num)
                    if grad is not None and ref_grad is not None:
                        payment = self.payment_function(grad, ref_grad)
                payments.append(payment)

            seller_history['payment_received'] = payments
            return seller_history

        return seller_history  # Should not be reached

    def _simulate_binary_payment(self, seller_history: pd.DataFrame) -> pd.DataFrame:
        """Pay-per-selection model: Fixed payment if selected."""
        base_payment = self.params.get('base_payment', 5.0)
        noise = self.params.get('noise_level', 0.5)

        is_selected = seller_history['assigned_weight'] > 0
        payments = is_selected * base_payment + np.random.normal(0, noise, len(seller_history))
        seller_history['payment_received'] = np.maximum(0, payments)
        return seller_history

    def _simulate_proportional_payment(self, seller_history: pd.DataFrame) -> pd.DataFrame:
        """Proportional-to-contribution model: Payment scales with weight."""
        scale_factor = self.params.get('scale_factor', 20.0)
        noise = self.params.get('noise_level', 0.5)

        payments = seller_history['assigned_weight'] * scale_factor + np.random.normal(0, noise, len(seller_history))
        seller_history['payment_received'] = np.maximum(0, payments)
        return seller_history

    def _calculate_quality_based_payment(self, individual_grad: np.ndarray, reference_grad: np.ndarray) -> float:
        """Quality-based model: Payment based on gradient similarity."""
        base_rate = self.params.get('base_rate', 2.0)
        bonus = self.params.get('similarity_bonus', 10.0)

        # This is a placeholder for your actual cosine_similarity function
        similarity = cosine_similarity(individual_grad, reference_grad)
        similarity = 0 if np.isnan(similarity) else similarity

        payment = base_rate + (bonus * max(0, similarity))
        return payment
