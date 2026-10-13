import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def group_outcome_table(y_true, sensitive):
    frame = pd.DataFrame({"outcome": y_true, "group": sensitive}).dropna()
    summary = frame.groupby("group")["outcome"].agg(["count", "mean", "sum"])
    return summary.rename(columns={"mean": "selection_rate", "sum": "originated"})


def chi_square_test(y_true, sensitive):
    table = pd.crosstab(pd.Series(sensitive, name="group"), pd.Series(y_true, name="outcome"))
    chi2, p, dof, expected = chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof, "observed": table, "expected": expected}


def parity_metrics(predictions, sensitive, reference_group):
    frame = pd.DataFrame({"prediction": predictions, "group": sensitive}).dropna()
    rates = frame.groupby("group")["prediction"].mean()
    if reference_group not in rates.index:
        raise ValueError(f"Reference group '{reference_group}' not found")
    ref = rates.loc[reference_group]
    return pd.DataFrame({
        "selection_rate": rates,
        "statistical_parity_difference": rates - ref,
        "disparate_impact_ratio": np.where(ref == 0, np.nan, rates / ref),
    })
