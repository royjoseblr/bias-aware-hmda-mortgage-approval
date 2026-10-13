import pandas as pd
from src.data import prepare_outcome


def test_prepare_outcome_filters_and_codes():
    df = pd.DataFrame({"action_taken": [1, 3, 2, 1]})
    out = prepare_outcome(df)
    assert out["action_taken"].tolist() == [1, 3, 1]
    assert out["outcome"].tolist() == [1, 0, 1]
