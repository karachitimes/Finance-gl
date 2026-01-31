
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, value_col: str, *, z: float = 3.5, contamination: float | None = None):
    """
    Pure-Python anomaly detection using robust MAD z-score.
    Adds: is_anomaly
    Parameters:
      z: threshold (higher = fewer anomalies)
      contamination: kept for compatibility (ignored in MAD approach)
    """
    out = df.copy()
    s = pd.to_numeric(out.get(value_col), errors="coerce").fillna(0.0).astype(float)

    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))

    if mad == 0:
        # fallback to standard deviation if MAD collapses
        std = float(s.std(ddof=0))
        if std == 0:
            out["is_anomaly"] = False
            return out
        score = (s - s.mean()) / std
    else:
        # 0.6745 makes MAD comparable to std under normal distribution
        score = 0.6745 * (s - med) / mad

    out["is_anomaly"] = score.abs() >= float(z)
    return out
