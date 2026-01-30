import streamlit as st
import pandas as pd
import numpy as np

def show_df(df: pd.DataFrame, *, label_col: str | None = None, total_label: str = "TOTAL"):
    """Render a dataframe with totals row across numeric columns."""
    if df is None or df.empty:
        st.info("No rows for selected filters.")
        return
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        total_row = {c: "" for c in out.columns}
        for c in num_cols:
            total_row[c] = float(pd.to_numeric(out[c], errors="coerce").fillna(0).sum())
        if label_col and label_col in out.columns:
            total_row[label_col] = total_label
        else:
            for c in out.columns:
                if c not in num_cols:
                    total_row[c] = total_label
                    break
        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(out, width="stretch")


def show_pivot(pivot: pd.DataFrame, total_label: str = "TOTAL"):
    if pivot is None or pivot.empty:
        st.info("No rows for selected filters.")
        return
    out = pivot.copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    out[total_label] = out.sum(axis=1)
    total_row = out.sum(axis=0).to_frame().T
    total_row.index = [total_label]
    out = pd.concat([out, total_row], axis=0)
    st.dataframe(out, width="stretch")

