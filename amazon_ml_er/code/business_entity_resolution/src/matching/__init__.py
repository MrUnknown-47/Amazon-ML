"""
Final Matching and Entity-Level Decisioning Package.
"""

from .features import (
    MatchingEntityProfile,
    extract_matching_features,
    extract_batch_matching_features,
    FEATURE_NAMES,
    NUM_MATCHING_FEATURES
)
from .dataset import (
    MatcherHardNegativeSampler,
    build_matching_dataset
)
from .models import FinalMatcherModel
from .calibration import (
    MatcherCalibrator,
    compute_calibration_diagnostics
)
from .entity_decision import (
    EntityDecisionEngine,
    compute_singleton_diagnostics
)
from .thresholding import search_optimal_threshold
from .conflict_resolution import resolve_target_conflicts_greedy
from .evaluation import evaluate_matcher_predictions
