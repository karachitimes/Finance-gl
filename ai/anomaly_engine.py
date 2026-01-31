
import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, value_col: str, *, z: float = 3.5, contamination: float | None = None):
    """Dependency-free anomaly detection using robust MAD z-score."""
    out = df.copy()
    s = pd.to_numeric(out.get(value_col), errors="coerce").fillna(0.0).astype(float)

    med = float(np.median(s)) if len(s) else 0.0
    mad = float(np.median(np.abs(s - med))) if len(s) else 0.0

    if mad == 0:
        std = float(s.std(ddof=0))
        if std == 0 or np.isnan(std):
            out["is_anomaly"] = False
            return out
        score = (s - float(s.mean())) / std
    else:
        score = 0.6745 * (s - med) / mad

    out["is_anomaly"] = score.abs() >= float(z)
    return out
