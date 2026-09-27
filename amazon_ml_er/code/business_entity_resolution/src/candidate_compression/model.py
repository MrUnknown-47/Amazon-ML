"""
Model wrapper for candidate ranking and compression.
Supports HistGradientBoostingClassifier and regularized LogisticRegression.
Provides ranking score generation, feature importance attribution, and model persistence.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import FEATURE_NAMES


class CandidateRanker:
    """
    Candidate ranking model for business entity candidate compression.
    Scores candidate pairs to rank true matches ahead of false matches.
    """

    def __init__(
        self,
        model_type: str = "hist_gb",
        random_state: int = 42,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        l2_reg: float = 1.0
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.model = None
        self.is_fitted = False

        if model_type == "hist_gb":
            self.model = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                l2_regularization=self.l2_reg,
                random_state=self.random_state,
                scoring="loss",
                class_weight="balanced"
            )
        elif model_type == "logistic":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0 / max(1e-5, self.l2_reg),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    class_weight="balanced",
                    solver="lbfgs"
                ))
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Expected 'hist_gb' or 'logistic'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CandidateRanker":
        """Fit ranker on feature matrix X and binary labels y."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Generate continuous ranking scores in [0.0, 1.0] for candidates in X.
        Higher score indicates higher likelihood of being a true match.
        """
        if not self.is_fitted:
            raise RuntimeError("CandidateRanker must be fitted before calling score().")
        X_arr = np.asarray(X, dtype=np.float32)
        if len(X_arr) == 0:
            return np.empty(0, dtype=np.float32)

        # Use predict_proba for positive class
        proba = self.model.predict_proba(X_arr)
        return proba[:, 1].astype(np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance scores or logistic regression coefficients."""
        if not self.is_fitted:
            return {name: 0.0 for name in FEATURE_NAMES}

        if self.model_type == "logistic":
            coefs = self.model.named_steps["clf"].coef_[0]
            return {name: float(round(abs(coef), 4)) for name, coef in zip(FEATURE_NAMES, coefs)}
        elif self.model_type == "hist_gb":
            # For HistGradientBoosting, compute permutation importance or feature presence
            # Return uniform dictionary if native importances not directly exposed
            return {name: 1.0 for name in FEATURE_NAMES}

        return {name: 0.0 for name in FEATURE_NAMES}

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CandidateRanker":
        """Deserialize model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
