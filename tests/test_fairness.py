import numpy as np
from src.fairness import parity_metrics


def test_parity_metrics_reference():
    result = parity_metrics(np.array([1, 1, 0, 0]), np.array(['A','A','B','B']), 'A')
    assert result.loc['A', 'statistical_parity_difference'] == 0
    assert result.loc['B', 'disparate_impact_ratio'] == 0
