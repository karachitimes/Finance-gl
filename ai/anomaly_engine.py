import pandas as pd

def _zscore_flags(s: pd.Series, z: float = 3.5) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series([False] * len(s), index=s.index)
    zscores = (s - mu) / sigma
    return zscores.abs() >= z

def detect_anomalies(df: pd.DataFrame, value_col: str, contamination: float = 0.02, z: float = 3.5):
    """
    Tries IsolationForest if sklearn exists; otherwise uses robust z-score fallback.
    Returns df with: is_anomaly, anomaly_method
    """
    out = df.copy()
    out[value_col] = pd.to_numeric(out.get(value_col), errors="coerce").fillna(0.0)

    try:
        from sklearn.ensemble import IsolationForest  # optional dependency
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(out[[value_col]])
        out["is_anomaly"] = (preds == -1)
        out["anomaly_method"] = "isolation_forest"
        return out
    except Exception:
        out["is_anomaly"] = _zscore_flags(out[value_col], z=z)
        out["anomaly_method"] = "zscore"
        return out
