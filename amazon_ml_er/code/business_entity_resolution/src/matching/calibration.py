"""
Probability Calibration and Reliability Diagnostics for Final Matcher.
Supports Platt/Sigmoid and Isotonic calibration fitted strictly on training/tuning data.
Computes Brier score, log-loss, and calibration curve bin diagnostics.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class MatcherCalibrator:
    """
    Fits and applies probability calibration to raw matcher scores.
    Strictly isolated: Must be trained on TUNE or TRAIN data only.
    """

    def __init__(self, method: str = "sigmoid"):
        """
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        """
        self.method = method
        self.is_fitted = False
        self.calibrator = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "MatcherCalibrator":
        """Fit calibration curve mapping raw scores [0, 1] to empirical probabilities."""
        if len(raw_scores) == 0:
            raise ValueError("Cannot fit calibrator on empty scores.")

        scores_reshaped = raw_scores.reshape(-1, 1)

        if self.method == "sigmoid":
            # Platt scaling via 1D logistic regression
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(scores_reshaped, y_true)
            self.calibrator = lr
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_scores, y_true)
            self.calibrator = iso
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to raw scores."""
        if not self.is_fitted:
            return raw_scores
        if len(raw_scores) == 0:
            return np.array([], dtype=np.float32)

        if self.method == "sigmoid":
            probs = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            return probs.astype(np.float32)
        elif self.method == "isotonic":
            probs = self.calibrator.predict(raw_scores)
            return np.clip(probs, 0.0, 1.0).astype(np.float32)
        return raw_scores


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute reliability curve, Brier score, and log-loss for probability evaluation.
    """
    if len(y_true) == 0:
        return {"brier_score": 0.0, "log_loss": 0.0, "bins": []}

    brier = float(round(brier_score_loss(y_true, probs), 5))
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1 - eps)
    loss = float(round(log_loss(y_true, clipped_probs), 5))

    # Bin diagnostics
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(round(np.mean(probs[mask]), 4))
            empirical_pos = float(round(np.mean(y_true[mask]), 4))
        else:
            mean_pred = float(round((low + high) / 2.0, 4))
            empirical_pos = 0.0

        bins_data.append({
            "bin_range": f"[{low:.2f}, {high:.2f}]",
            "count": count,
            "mean_pred_prob": mean_pred,
            "empirical_positive_rate": empirical_pos,
            "calibration_error": float(round(abs(mean_pred - empirical_pos), 4))
        })

    # Expected Calibration Error (ECE)
    total_samples = len(y_true)
    ece = float(round(sum(b["count"] * b["calibration_error"] for b in bins_data) / total_samples, 5))

    return {
        "brier_score": brier,
        "log_loss": loss,
        "expected_calibration_error": ece,
        "bins": bins_data
    }
