import streamlit as st
import pandas as pd
from datetime import date

from db import run_df
from utils import show_df


PAR_STREAMS = ("PAR", "WAR", "AMC", "AGR")


def _money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "0"


def _aging_bucket(days: float):
    if days is None or pd.isna(days):
        return "unknown"
    d = float(days)
    if d <= 30:
        return "0-30"
    if d <= 60:
        return "31-60"
    if d <= 90:
        return "61-90"
    if d <= 180:
        return "91-180"
    return "180+"


def _get_columns(engine, rel: str) -> set[str]:
    """
    Reliable even if information_schema is blocked.
    Uses SELECT * LIMIT 0 to fetch column names.
    """
    try:
        df = run_df(engine, f"select * from {rel} limit 0", {}, rel=rel)
        if df is None:
            return set()
        return set([str(c) for c in df.columns])
    except Exception:
        return set()


def _probe_columns(engine, rel: str, candidates: list[str]) -> set[str]:
    found: set[str] = set()
    for c in candidates:
        sql = f'select "{c}" from {rel} limit 1'
        try:
            _ = run_df(engine, sql, {}, rel=rel)
            found.add(c)
        except Exception:
            pass
    return found


def _col_or_literal(cols: set[str], col: str, literal_sql: str):
    return f'"{col}"' if col in cols else f"{literal_sql} as {col}"


def _billing_stream_from_head(head: str) -> str:
    h = (head or "").upper()
    for s in PAR_STREAMS:
        if s in h:
            return s
    return "OTHER"


def render_billing_tab(engine, f, *, rel: str):
    st.subheader("🧾 Billing Reconciliation Engine")
    st.caption(
        "Bills issued are tracked in debit_payment; receipts in credit_deposit. "
        "PAR/WAR/AMC/AGR are treated as customer-ledger streams (keyed by account + subhead1). "
        "Government Grant is reported separately from ordinary income."
    )

    cols = _get_columns(engine, rel)
    if not cols:
        st.warning("Could not read columns via LIMIT 0. Falling back to probing.")
        cols = _probe_columns(
            engine,
            rel,
            [
                "date",
                "reconcile",
                "head_name",
                "head_no",
                "subhead1",
                "account",
                "bill_no",
                "folio_chq_no",
                "bank",
                "debit_payment",
                "credit_deposit",
                "description",
                "ref",
                "pay_to",
                "net_flow",
            ],
        )

    has_reconcile = "reconcile" in cols

    # ---------------------------------------------------------
    # ✅ GLOBAL FILTERING STYLE (same as rest of app)
    # ---------------------------------------------------------
    # Preferred: reuse filters computed in app.py (consistent across tabs)
    where_sql = f.get("where_sql")
    params = f.get("params")

    # Fallback: if billing is used standalone, build minimal date/bank filter
    if not where_sql or not params:
        params = {"df": f.get("df"), "dt": f.get("dt")}
        # choose a date column
        if "date" in cols:
            date_col = '"date"'
        elif "reconcile" in cols:
            date_col = '"reconcile"'
        else:
            st.error(
                "Billing tab cannot run because neither 'date' nor 'reconcile' exists in this relation."
            )
            return
        where = [f"{date_col} between :df and :dt"]
        if f.get("bank") and f["bank"] != "ALL" and "bank" in cols:
            where.append('"bank" = :bank')
            params["bank"] = f["bank"]
        where_sql = " and ".join(where) if where else "1=1"

    # Optional: Bill search (Billing-only filter)
    bill_like = st.text_input("Filter bill_no contains", value="")
    where_bill = ""
    if bill_like.strip():
        if "bill_no" in cols:
            where_bill = 'and coalesce("bill_no", \'\') ilike :bill_like'
            params = dict(params)
            params["bill_like"] = f"%{bill_like.strip()}%"
        else:
            st.info("bill_no filter ignored because column 'bill_no' is not present in this relation.")

    # Date basis choice (only affects aging base field, not core filtering)
    date_basis = st.radio(
        "Date basis for aging",
        ["Accounting date (date)"] + (["Bank reconciliation (reconcile)"] if has_reconcile else []),
        horizontal=True,
    )
    if not has_reconcile:
        st.info("Bank reconciliation not available (column 'reconcile' not found).")

    # ---------------------------------------------------------
    # Pull ledger rows (bounded)
    # ---------------------------------------------------------
    sel = []
    sel.append(_col_or_literal(cols, "date", "null::date"))
    if has_reconcile:
        sel.append(_col_or_literal(cols, "reconcile", "null::timestamp"))
    sel.append(_col_or_literal(cols, "head_name", "''"))
    sel.append(_col_or_literal(cols, "head_no", "''"))
    sel.append(_col_or_literal(cols, "subhead1", "''"))
    sel.append(_col_or_literal(cols, "account", "''"))
    sel.append(_col_or_literal(cols, "bill_no", "''"))
    sel.append(_col_or_literal(cols, "folio_chq_no", "''"))
    sel.append(_col_or_literal(cols, "bank", "''"))
    if "ref" in cols:
        sel.append(_col_or_literal(cols, "ref", "''"))
    if "pay_to" in cols:
        sel.append(_col_or_literal(cols, "pay_to", "''"))
    if "description" in cols:
        sel.append(_col_or_literal(cols, "description", "''"))
    sel.append(_col_or_literal(cols, "debit_payment", "0::numeric"))
    sel.append(_col_or_literal(cols, "credit_deposit", "0::numeric"))

    sql_sample = f"""
        select {", ".join(sel)}
        from {rel}
        where {where_sql}
        {where_bill}
        limit 5000
    """
    df0 = run_df(engine, sql_sample, params, rel=rel)

    if df0 is None or df0.empty:
        st.warning("No billing ledger rows found under current filters/date range.")
        return

    # Normalize
    df0["debit_payment"] = pd.to_numeric(df0.get("debit_payment"), errors="coerce").fillna(0.0)
    df0["credit_deposit"] = pd.to_numeric(df0.get("credit_deposit"), errors="coerce").fillna(0.0)
    df0["head_name"] = df0.get("head_name", "").astype(str)
    df0["subhead1"] = df0.get("subhead1", "").astype(str)
    df0["account"] = df0.get("account", "").astype(str)
    df0["bill_no"] = df0.get("bill_no", "").astype(str)
    df0["folio_chq_no"] = df0.get("folio_chq_no", "").astype(str)
    df0["bank"] = df0.get("bank", "").astype(str)

    df0["billing_stream"] = df0["head_name"].apply(_billing_stream_from_head)
    df0["is_grant"] = df0["subhead1"].str.contains("Government Grant", case=False, na=False)
    df0["is_indirect_income"] = df0["subhead1"].str.contains(r"Income \(Indirect/Opr", case=False, na=False)

    df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
    if has_reconcile and ("reconcile" in df0.columns):
        df0["reconcile"] = pd.to_datetime(df0["reconcile"], errors="coerce")

    # ---------------------------------------------------------
    # Reconciliation key selection
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 🔗 Reconciliation key")

    key_mode = st.selectbox(
        "How to reconcile (logical bill key)",
        ["bill_no only", "bill_no + account", "account-ledger (account + subhead1)"],
        index=1,
    )

    if key_mode == "account-ledger (account + subhead1)":
        df0["bill_key"] = df0["account"].str.strip() + " | " + df0["subhead1"].str.strip()
    elif key_mode == "bill_no + account":
        df0["bill_key"] = df0["bill_no"].str.strip() + " | " + df0["account"].str.strip()
    else:
        df0["bill_key"] = df0["bill_no"].str.strip()

    # ---------------------------------------------------------
    # Grants separated
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 🏷 Income classification (Grants vs Ordinary)")

    receipts_total = float(df0.loc[df0["credit_deposit"] > 0, "credit_deposit"].sum())
    grant_receipts = float(df0.loc[(df0["credit_deposit"] > 0) & (df0["is_grant"]), "credit_deposit"].sum())
    grant_ratio = (grant_receipts / receipts_total * 100.0) if receipts_total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Receipts total", _money(receipts_total))
    c2.metric("Grant receipts", _money(grant_receipts))
    c3.metric("Grant dependency %", f"{grant_ratio:.1f}%")

    # ---------------------------------------------------------
    # Indirect income rollups
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 🧩 Indirect/Opr income rollups (subhead1 → head_name → account)")

    df_ind = df0[(df0["is_indirect_income"]) & (df0["credit_deposit"] > 0)].copy()
    if df_ind.empty:
        st.info("No rows found under subhead1 = Income (Indirect/Opr.) within current filters.")
    else:
        roll2 = (
            df_ind.groupby(["subhead1", "head_name"], as_index=False)
            .agg(receipts=("credit_deposit", "sum"), rows=("credit_deposit", "size"))
            .sort_values("receipts", ascending=False)
        )
        show_df(roll2.head(200))

        roll3 = (
            df_ind.groupby(["subhead1", "head_name", "account"], as_index=False)
            .agg(receipts=("credit_deposit", "sum"), rows=("credit_deposit", "size"))
            .sort_values("receipts", ascending=False)
        )
        show_df(roll3.head(300))

    # ---------------------------------------------------------
    # Ledger reconciliation (issued vs received)
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 📚 Billing ledger reconciliation (issued vs received vs outstanding)")

    df_led = df0[(df0["debit_payment"] != 0) | (df0["credit_deposit"] != 0)].copy()
    df_led["bill_key_routed"] = df_led["bill_key"]

    # Always route PAR/WAR/AMC/AGR to account|subhead1
    par_mask = df_led["billing_stream"].isin(PAR_STREAMS)
    df_led.loc[par_mask, "bill_key_routed"] = (
        df_led.loc[par_mask, "account"].str.strip() + " | " + df_led.loc[par_mask, "subhead1"].str.strip()
    )

    agg = df_led.groupby(["bill_key_routed"], as_index=False).agg(
        issued=("debit_payment", "sum"),
        received=("credit_deposit", "sum"),
        last_activity=("date", "max"),
        last_reconcile=("reconcile", "max") if has_reconcile else ("date", "max"),
        any_bill_no=("bill_no", "max"),
        any_account=("account", "max"),
        any_subhead1=("subhead1", "max"),
    )
    agg["outstanding"] = agg["issued"] - agg["received"]

    # aging base
    if ("Bank reconciliation" in date_basis) and has_reconcile:
        base_dt = pd.to_datetime(agg["last_reconcile"], errors="coerce")
    else:
        base_dt = pd.to_datetime(agg["last_activity"], errors="coerce")

    today = pd.to_datetime(date.today())
    agg["days_since"] = (today - base_dt).dt.days
    agg["aging_bucket"] = agg["days_since"].apply(_aging_bucket)

    show_df(agg.sort_values("outstanding", ascending=False).head(200))

    # ---------------------------------------------------------
    # ✅ Drill-down (underlying rows)
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 🔎 Drill-down: open a bill / ledger")

    if not agg.empty:
        selected_key = st.selectbox(
            "Drill into bill / ledger",
            options=agg["bill_key_routed"].astype(str).tolist(),
            index=0,
        )

        if selected_key:
            key = selected_key.strip()

            drill_where_sql = where_sql
            drill_params = dict(params)

            if " | " in key:
                acct, subh = [x.strip() for x in key.split(" | ", 1)]
                if "account" in cols:
                    drill_where_sql += ' and "account" = :acct'
                    drill_params["acct"] = acct
                if "subhead1" in cols:
                    drill_where_sql += ' and "subhead1" = :subh'
                    drill_params["subh"] = subh
            else:
                if "bill_no" in cols:
                    drill_where_sql += ' and "bill_no" = :bill_no'
                    drill_params["bill_no"] = key
                else:
                    st.warning("Cannot drill by bill_no because column 'bill_no' is not present in this relation.")
                    drill_where_sql = None

            if drill_where_sql:
                detail_sel = []
                detail_sel.append(_col_or_literal(cols, "date", "null::date"))
                if has_reconcile:
                    detail_sel.append(_col_or_literal(cols, "reconcile", "null::timestamp"))
                detail_sel.append(_col_or_literal(cols, "bank", "''"))
                detail_sel.append(_col_or_literal(cols, "head_name", "''"))
                detail_sel.append(_col_or_literal(cols, "head_no", "''"))
                detail_sel.append(_col_or_literal(cols, "subhead1", "''"))
                detail_sel.append(_col_or_literal(cols, "account", "''"))
                detail_sel.append(_col_or_literal(cols, "bill_no", "''"))
                detail_sel.append(_col_or_literal(cols, "folio_chq_no", "''"))
                if "ref" in cols:
                    detail_sel.append(_col_or_literal(cols, "ref", "''"))
                if "pay_to" in cols:
                    detail_sel.append(_col_or_literal(cols, "pay_to", "''"))
                if "description" in cols:
                    detail_sel.append(_col_or_literal(cols, "description", "''"))
                detail_sel.append(_col_or_literal(cols, "debit_payment", "0::numeric"))
                detail_sel.append(_col_or_literal(cols, "credit_deposit", "0::numeric"))

                sql_drill = f"""
                    select {", ".join(detail_sel)}
                    from {rel}
                    where {drill_where_sql}
                    order by "date" desc nulls last
                    limit 5000
                """
                df_drill = run_df(engine, sql_drill, drill_params, rel=rel)
                show_df(df_drill)

    # ---------------------------------------------------------
    # Aging totals
    # ---------------------------------------------------------
    st.markdown("### Aging bucket totals (outstanding)")
    buckets = (
        agg.groupby("aging_bucket", as_index=False)
        .agg(items=("bill_key_routed", "count"), outstanding=("outstanding", "sum"))
        .sort_values("aging_bucket")
    )
    show_df(buckets)

    # ---------------------------------------------------------
    # Controls & exceptions
    # ---------------------------------------------------------
    st.divider()
    st.markdown("## 🧾 Controls & exceptions")

    unapplied = df0[(df0["credit_deposit"] > 0) & (df0["bill_no"].str.strip() == "")].copy()
    st.markdown("### Receipts without bill_no reference")
    show_df(unapplied.sort_values("date", ascending=False).head(300))

    missing_receipt_no = df0[(df0["credit_deposit"] > 0) & (df0["folio_chq_no"].str.strip() == "")].copy()
    st.markdown("### Receipts missing receipt no (folio_chq_no)")
    show_df(missing_receipt_no.sort_values("date", ascending=False).head(300))

    st.divider()
    st.markdown("## 🏦 Deposit routing (bank)")

    by_bank = (
        df0[df0["credit_deposit"] > 0]
        .groupby(["bank"], as_index=False)
        .agg(
            receipts=("credit_deposit", "sum"),
            rows=("credit_deposit", "size"),
            with_receipt_no=("folio_chq_no", lambda s: (s.astype(str).str.strip() != "").sum()),
        )
        .sort_values("receipts", ascending=False)
    )
    show_df(by_bank.head(50))

    st.markdown("### Potential bank posting mismatch (same routed key across multiple banks)")
    tmp = df0[df0["credit_deposit"] > 0].copy()
    tmp["bill_key_routed"] = tmp["bill_key"]
    par_mask2 = tmp["billing_stream"].isin(PAR_STREAMS)
    tmp.loc[par_mask2, "bill_key_routed"] = (
        tmp.loc[par_mask2, "account"].str.strip() + " | " + tmp.loc[par_mask2, "subhead1"].str.strip()
    )

    bank_mix = tmp.groupby(["bill_key_routed"], as_index=False).agg(
        banks=("bank", lambda s: len(set([x for x in s.astype(str) if x.strip()]))),
        receipts=("credit_deposit", "sum"),
        last_date=("date", "max"),
    )
    bank_mix = bank_mix[bank_mix["banks"] >= 2].sort_values("receipts", ascending=False).head(200)
    show_df(bank_mix)
