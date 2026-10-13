import numpy as np


def inverse_frequency_weights(sensitive):
    values, counts = np.unique(sensitive, return_counts=True)
    weight_map = {v: len(sensitive) / (len(values) * c) for v, c in zip(values, counts)}
    return np.array([weight_map[v] for v in sensitive])


def apply_group_thresholds(scores, sensitive, thresholds, default=0.5):
    return np.array([int(s >= thresholds.get(g, default)) for s, g in zip(scores, sensitive)])
