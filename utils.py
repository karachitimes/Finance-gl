import io
import streamlit as st
import pandas as pd
import numpy as np

from db import run_df  # ✅ required for drill_panel


def show_df(
    df: pd.DataFrame,
    *,
    label_col: str | None = None,
    total_label: str = "TOTAL",
    max_rows: int = 5000,
    show_export: bool = True
):
    """
    Render a dataframe with an optional totals row across numeric columns.
    Also supports truncation + Excel export safely.
    """
    if df is None or df.empty:
        st.info("No rows for selected filters.")
        return

    # Work on a copy for display
    out = df.copy()

    # Add totals row across numeric columns
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        total_row = {c: "" for c in out.columns}
        for c in num_cols:
            total_row[c] = float(pd.to_numeric(out[c], errors="coerce").fillna(0).sum())

        # Put TOTAL in label_col if provided; else first non-numeric column
        if label_col and label_col in out.columns:
            total_row[label_col] = total_label
        else:
            for c in out.columns:
                if c not in num_cols:
                    total_row[c] = total_label
                    break

        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    # Truncate for display (but do not truncate the original df used for export)
    truncated = out
    if len(out) > max_rows:
        st.warning(f"Showing first {max_rows:,} rows (table truncated).")
        truncated = out.head(max_rows)

    st.dataframe(truncated, use_container_width=True)

    # Export (export the ORIGINAL df, not the totals/truncated version, unless you prefer otherwise)
    if show_export:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            buf.seek(0)
            st.download_button(
                "Export Excel",
                data=buf,
                file_name="data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.info("Export not available for this table.")
            st.caption(str(e))


def drill_panel(engine, rel: str, base_where: list[str], base_params: dict, *, title: str, drill_col: str, drill_value):
    """
    Generic drill-down panel: adds a drill filter to the base WHERE clauses and shows underlying rows.
    """
    st.markdown(f"### 🔎 Drill-down: {title}")

    where = list(base_where) + [f'"{drill_col}" = :drill_value']
    params = dict(base_params)
    params["drill_value"] = drill_value

    sql = f"""
        select *
        from {rel}
        where {' and '.join(where) if where else '1=1'}
        order by "date" desc nulls last
        limit 5000
    """
    df = run_df(engine, sql, params, rel=rel)
    show_df(df, show_export=True)


def show_pivot(pivot: pd.DataFrame, total_label: str = "TOTAL"):
    """
    Render pivot with row/col totals for numeric values.
    """
    if pivot is None or pivot.empty:
        st.info("No rows for selected filters.")
        return

    out = pivot.copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    out[total_label] = out.sum(axis=1)

    total_row = out.sum(axis=0).to_frame().T
    total_row.index = [total_label]

    out = pd.concat([out, total_row], axis=0)
    st.dataframe(out, use_container_width=True)
