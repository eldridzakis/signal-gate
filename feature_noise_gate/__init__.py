"""Sampling Toolbox: Statistical rigor for feature evaluation & selection.

Quantifying the Winner's Curse, estimating noise thresholds, and de-biasing
automated feature engineering.
"""

from .metrics import (
    entropy,
    rig,
    rig_all_columns,
    fast_contingency_matrix,
    fast_mutual_info_from_contingency
)
from .threshold import (
    FeatureNoiseThreshold,
    estimate_noise_threshold,
    binary_entropy,
    target_entropy_from_weights
)
from .permutation import (
    PermutationObject
)
from .debias import (
    FeatureDebiaser
)

__version__ = "1.0.0"
__all__ = [
    "entropy",
    "rig",
    "rig_all_columns",
    "fast_contingency_matrix",
    "fast_mutual_info_from_contingency",
    "FeatureNoiseThreshold",
    "estimate_noise_threshold",
    "binary_entropy",
    "target_entropy_from_weights",
    "PermutationObject",
    "FeatureDebiaser",
]
