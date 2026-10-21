"""Backward-compatibility module forwarding to feature_noise_gate."""

from feature_noise_gate import (
    entropy,
    rig,
    rig_all_columns,
    fast_contingency_matrix,
    fast_mutual_info_from_contingency,
    FeatureNoiseThreshold,
    estimate_noise_threshold,
    binary_entropy,
    target_entropy_from_weights,
    PermutationObject,
    FeatureDebiaser,
    __version__,
)

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
