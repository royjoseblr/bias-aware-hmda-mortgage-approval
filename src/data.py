from pathlib import Path
import pandas as pd


def load_hmda(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw HMDA file not found: {p}")
    return pd.read_csv(p, low_memory=False)


def prepare_outcome(df: pd.DataFrame) -> pd.DataFrame:
    if "action_taken" not in df.columns:
        raise ValueError("Required column 'action_taken' is missing")
    out = df[df["action_taken"].isin([1, 3])].copy()
    out["outcome"] = (out["action_taken"] == 1).astype(int)
    return out


def select_columns(df, numeric, categorical, audit, exclusions):
    exclusions = set(exclusions)
    keep = [c for c in numeric + categorical + audit + ["outcome"] if c in df.columns and c not in exclusions]
    return df.loc[:, keep].copy()
