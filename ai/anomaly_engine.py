
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, value_col, contamination=0.02):
    model = IsolationForest(contamination=contamination, random_state=42)
    X = df[[value_col]].fillna(0)
    df["anomaly_flag"] = model.fit_predict(X)
    df["is_anomaly"] = df["anomaly_flag"] == -1
    return df
