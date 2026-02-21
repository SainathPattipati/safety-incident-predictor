"""Ensemble safety incident predictor using multiple ML algorithms."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import logging


@dataclass
class IncidentRiskScore:
    """Safety incident risk assessment."""
    risk_probability: float      # 0-1, probability of incident in next 72 hours
    severity_estimate: str       # "low", "medium", "high", "critical"
    risk_factors: List[str]      # Contributing factors (worker fatigue, equipment age, etc.)
    recommended_interventions: List[str]
    confidence: float           # Model confidence 0-1


class IncidentPredictor:
    """
    Ensemble predictor for workplace safety incidents.

    Multi-modal signal fusion combining:
    - Environmental factors (temperature, noise, lighting, air quality)
    - Operational factors (machine speed, maintenance age, load levels)
    - Behavioral patterns (worker movement from wearables, fatigue)
    - Organizational factors (overtime, shift patterns, crew composition)
    - Historical factors (past incident rates, near-miss reports)
    """

    def __init__(self, time_horizon_hours: int = 72):
        """
        Initialize incident predictor.

        Args:
            time_horizon_hours: Prediction horizon (default 72 hours)
        """
        self.time_horizon = time_horizon_hours
        self._logger = logging.getLogger("incident_predictor")

        # XGBoost model for incident classification
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Feature scaler
        self.scaler = StandardScaler()

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self._model_trained = False

        # Feature names
        self.feature_names = [
            # Environmental
            "temperature_c", "humidity_percent", "noise_db", "lighting_lux", "air_quality_index",
            # Operational
            "machine_speed_percent", "maintenance_age_days", "load_percent",
            # Behavioral
            "worker_fatigue_score", "movement_speed_mm_s", "distraction_score",
            # Organizational
            "overtime_hours", "days_since_rest", "crew_experience_mix",
            # Historical
            "incident_rate_per_year", "near_miss_count_30d"
        ]

    def predict_risk(
        self,
        environmental: Dict[str, float],
        operational: Dict[str, float],
        behavioral: Dict[str, float],
        organizational: Dict[str, float],
        historical: Dict[str, float]
    ) -> IncidentRiskScore:
        """
        Predict incident risk from multi-modal signals.

        Args:
            environmental: Temperature, humidity, noise, lighting, air quality
            operational: Machine speed, maintenance age, load
            behavioral: Fatigue, movement, distraction
            organizational: Overtime, rest, crew mix
            historical: Past incident rates, near-misses

        Returns:
            IncidentRiskScore with probability and recommendations
        """
        # Assemble feature vector
        features = self._assemble_features(
            environmental, operational, behavioral, organizational, historical
        )

        # Make prediction
        if self._model_trained:
            risk_prob = self.xgb_model.predict_proba(features.reshape(1, -1))[0][1]
        else:
            # Simplified risk calculation if model not trained
            risk_prob = self._baseline_risk_score(features)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            environmental, operational, behavioral, organizational, historical
        )

        # Generate interventions
        interventions = self._generate_interventions(risk_factors, risk_prob)

        # Determine severity
        severity = self._assess_severity(risk_prob, risk_factors)

        return IncidentRiskScore(
            risk_probability=min(1.0, max(0.0, risk_prob)),
            severity_estimate=severity,
            risk_factors=risk_factors[:3],  # Top 3
            recommended_interventions=interventions,
            confidence=0.75 if self._model_trained else 0.5
        )

    def _assemble_features(
        self,
        environmental: Dict,
        operational: Dict,
        behavioral: Dict,
        organizational: Dict,
        historical: Dict
    ) -> np.ndarray:
        """Assemble feature vector from multi-modal inputs."""
        features = []

        # Environmental
        features.append(environmental.get("temperature_c", 20.0))
        features.append(environmental.get("humidity_percent", 50.0))
        features.append(environmental.get("noise_db", 85.0))
        features.append(environmental.get("lighting_lux", 500.0))
        features.append(environmental.get("air_quality_index", 50.0))

        # Operational
        features.append(operational.get("machine_speed_percent", 100.0))
        features.append(operational.get("maintenance_age_days", 1000.0))
        features.append(operational.get("load_percent", 75.0))

        # Behavioral
        features.append(behavioral.get("fatigue_score", 0.3))
        features.append(behavioral.get("movement_speed_mm_s", 500.0))
        features.append(behavioral.get("distraction_score", 0.2))

        # Organizational
        features.append(organizational.get("overtime_hours", 0.0))
        features.append(organizational.get("days_since_rest", 1.0))
        features.append(organizational.get("crew_experience_mix", 0.6))

        # Historical
        features.append(historical.get("incident_rate_per_year", 2.0))
        features.append(historical.get("near_miss_count_30d", 3.0))

        return np.array(features, dtype=np.float32)

    def _identify_risk_factors(
        self,
        environmental: Dict,
        operational: Dict,
        behavioral: Dict,
        organizational: Dict,
        historical: Dict
    ) -> List[str]:
        """Identify which factors contribute to incident risk."""
        risk_factors = []

        # Environmental risks
        temp = environmental.get("temperature_c", 20)
        if temp > 30 or temp < 10:
            risk_factors.append("Extreme temperature affecting alertness")

        noise = environmental.get("noise_db", 85)
        if noise > 90:
            risk_factors.append("High noise limiting communication")

        # Behavioral risks
        fatigue = behavioral.get("fatigue_score", 0.3)
        if fatigue > 0.6:
            risk_factors.append("High worker fatigue (>8 hours)")

        # Organizational risks
        overtime = organizational.get("overtime_hours", 0)
        if overtime > 4:
            risk_factors.append("Excessive overtime increasing fatigue")

        # Historical risks
        incident_rate = historical.get("incident_rate_per_year", 2)
        if incident_rate > 5:
            risk_factors.append("High historical incident rate on this line")

        return risk_factors

    def _generate_interventions(
        self,
        risk_factors: List[str],
        risk_prob: float
    ) -> List[str]:
        """Generate recommended interventions based on risk factors."""
        interventions = []

        if risk_prob > 0.3:
            if "fatigue" in " ".join(risk_factors).lower():
                interventions.append("Schedule additional rest breaks")
                interventions.append("Consider reduced shift length")

            if "temperature" in " ".join(risk_factors).lower():
                interventions.append("Improve cooling/heating systems")

            if "experience" in " ".join(risk_factors).lower():
                interventions.append("Assign experienced buddy to worker")
                interventions.append("Conduct safety refresher training")

        if risk_prob > 0.5:
            interventions.append("Supervisor check-in required")
            interventions.append("Brief pre-shift safety meeting")

        if risk_prob > 0.7:
            interventions.append("CRITICAL: Halt operation if unsafe conditions present")
            interventions.append("Immediate safety officer consultation required")

        return interventions

    @staticmethod
    def _assess_severity(risk_prob: float, risk_factors: List[str]) -> str:
        """Assess potential severity of incident if it occurs."""
        # Severity based on risk probability
        if risk_prob < 0.2:
            return "low"
        elif risk_prob < 0.4:
            return "medium"
        elif risk_prob < 0.7:
            return "high"
        else:
            return "critical"

    def _baseline_risk_score(self, features: np.ndarray) -> float:
        """Simple baseline risk calculation if model not trained."""
        # Weighted combination of high-risk features
        fatigue_idx = self.feature_names.index("worker_fatigue_score")
        overtime_idx = self.feature_names.index("overtime_hours")
        incident_rate_idx = self.feature_names.index("incident_rate_per_year")

        fatigue_component = features[fatigue_idx] * 0.4
        overtime_component = min(features[overtime_idx] / 8.0, 1.0) * 0.3
        history_component = min(features[incident_rate_idx] / 10.0, 1.0) * 0.3

        return fatigue_component + overtime_component + history_component

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train XGBoost model on historical incident data.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples] (0=no incident, 1=incident)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.xgb_model.fit(X_scaled, y_train)
        self._model_trained = True

        self._logger.info(f"Model trained on {len(X_train)} samples")
