
import pandas as pd

def correlation_root_cause(df, target_col):
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    return corr
