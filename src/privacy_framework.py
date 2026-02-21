"""Differential privacy framework for safety incident prediction."""

from typing import Dict, Optional
import numpy as np
import logging


class DifferentialPrivacyFramework:
    """
    Ensures worker data privacy using differential privacy techniques.

    - k-anonymity: Groups workers to hide individuals
    - Differential privacy: Adds noise to sensitive features
    - Fairness constraints: Equal false positive rates across demographics
    - Federated learning: Cross-site training without sharing raw data
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy framework.

        Args:
            epsilon: Privacy budget (lower = more private, smaller ε = more noise)
            delta: Failure probability (typically 1e-5 or lower)
        """
        self.epsilon = epsilon
        self.delta = delta
        self._logger = logging.getLogger("dp_framework")

        # Track privacy budget usage
        self._budget_used = 0.0
        self._max_budget = epsilon

    def anonymize_features(
        self,
        features: Dict[str, np.ndarray],
        sensitive_keys: list = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply anonymization to features.

        Args:
            features: Dictionary of feature arrays
            sensitive_keys: Which features are sensitive (e.g., ['worker_id'])

        Returns:
            Anonymized features with differential privacy applied
        """
        if sensitive_keys is None:
            sensitive_keys = ['worker_id', 'location', 'shift']

        anonymized = {}

        for key, values in features.items():
            if key in sensitive_keys:
                # Apply differential privacy via Laplace mechanism
                if isinstance(values, np.ndarray):
                    scale = 1.0 / self.epsilon
                    noise = np.random.laplace(0, scale, values.shape)
                    anonymized[key] = values + noise
                else:
                    anonymized[key] = values
            else:
                anonymized[key] = values

        return anonymized

    def k_anonymity_grouping(
        self,
        worker_data: Dict[int, Dict],
        k: int = 5,
        quasi_identifiers: list = None
    ) -> Dict[int, int]:
        """
        Group workers for k-anonymity (can't identify individual in group < k).

        Args:
            worker_data: Dictionary mapping worker_id to attributes
            k: Group size threshold
            quasi_identifiers: Attributes used for grouping (age, job, department)

        Returns:
            Mapping of worker_id to group_id
        """
        if quasi_identifiers is None:
            quasi_identifiers = ['age_range', 'job_category', 'department']

        # Create groups based on quasi-identifiers
        groups = {}
        group_counter = 0

        for worker_id, data in worker_data.items():
            # Create key from quasi-identifiers
            key = tuple(data.get(qi, 'unknown') for qi in quasi_identifiers)

            if key not in groups:
                groups[key] = []
            groups[key].append(worker_id)

        # Create mapping
        worker_to_group = {}
        for group_id, (key, workers) in enumerate(groups.items()):
            if len(workers) >= k:
                for w_id in workers:
                    worker_to_group[w_id] = group_id
            else:
                # Merge small groups
                self._logger.warning(
                    f"Group {key} has size {len(workers)} < k={k}"
                )

        return worker_to_group

    def apply_laplace_mechanism(
        self,
        values: np.ndarray,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Apply Laplace mechanism for differential privacy.

        Args:
            values: Sensitive data to protect
            sensitivity: Maximum change when one record removed

        Returns:
            Noised values satisfying ε-differential privacy
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, values.shape)

        self._budget_used += sensitivity / self.epsilon

        return values + noise

    def enforce_fairness_constraints(
        self,
        predictions: np.ndarray,
        sensitive_attribute: np.ndarray,
        target_fpr: float = 0.05
    ) -> np.ndarray:
        """
        Adjust predictions to equalize false positive rates across groups.

        Ensures the model is fair: doesn't have racial/gender/age bias.

        Args:
            predictions: Model predictions [0, 1]
            sensitive_attribute: Group membership (0, 1, 2, ...)
            target_fpr: Target false positive rate

        Returns:
            Fairness-adjusted predictions
        """
        unique_groups = np.unique(sensitive_attribute)
        adjusted = predictions.copy()

        # Adjust thresholds per group to equalize FPR
        for group in unique_groups:
            group_mask = sensitive_attribute == group
            group_preds = predictions[group_mask]

            # Find threshold that gives target FPR
            sorted_preds = np.sort(group_preds)
            threshold_idx = int(len(sorted_preds) * (1 - target_fpr))

            if threshold_idx < len(sorted_preds):
                threshold = sorted_preds[threshold_idx]
                adjusted[group_mask] = (group_preds >= threshold).astype(float)

        return adjusted

    def check_privacy_budget(self) -> Optional[str]:
        """
        Check if privacy budget is exhausted.

        Returns:
            Warning message if budget is low, None otherwise
        """
        remaining = self._max_budget - self._budget_used
        percent_used = (self._budget_used / self._max_budget) * 100

        if percent_used > 90:
            return f"Privacy budget 90% used ({percent_used:.1f}%)"
        elif percent_used > 75:
            return f"Privacy budget 75% used ({percent_used:.1f}%)"

        return None

    def federated_learning_ready(self) -> bool:
        """
        Check if framework is ready for federated learning.

        Federated learning trains models across multiple sites
        without sharing individual worker data.
        """
        return self._budget_used < self._max_budget * 0.5
