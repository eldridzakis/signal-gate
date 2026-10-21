"""Backward-compatibility module forwarding to the sampling_toolbox package."""

from sampling_toolbox import (
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
