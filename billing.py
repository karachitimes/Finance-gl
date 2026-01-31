
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
    if d <= 30: return "0-30"
    if d <= 60: return "31-60"
    if d <= 90: return "61-90"
    if d <= 180: return "91-180"
    return "180+"


def _try_has_column(engine, rel, f, col: str) -> bool:
    """Return True if relation has column; avoids hard-failing on unknown schemas."""
    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))
    sql = f'''select {col} from {rel} where {where_sql} limit 1'''
    try:
        _ = run_df(engine, sql, params, rel=rel)
        return True
    except Exception:
        return False


def _billing_stream_from_head(head: str) -> str:
    h = (head or "").upper()
    for s in PAR_STREAMS:
        if s in h:
            return s
    return "OTHER"


def _infer_key_mode_choice(df: pd.DataFrame) -> str:
    """Prefer account-ledger mode when PAR/WAR/AMC/AGR dominate."""
    if df is None or df.empty or "billing_stream" not in df.columns:
        return "auto"
    share = (df["billing_stream"].isin(PAR_STREAMS).mean()) if len(df) else 0.0
    return "account-ledger" if share >= 0.30 else "auto"


def render_billing_tab(engine, f, *, rel):
    st.subheader("🧾 Billing Reconciliation Engine")
    st.caption(
        "Bills issued are tracked in debit_payment; receipts in credit_deposit. "
        "PAR/WAR/AMC/AGR are treated as customer-ledger streams (keyed by account + subhead1). "
        "Government Grant is reported separately from ordinary income."
    )

    where_sql = f.get("where_sql", "1=1")
    params = dict(f.get("params", {}))

    # ---- Detect reconcile column for bank timeline ----
    has_reconcile = _try_has_column(engine, rel, f, "reconcile")
    date_basis = st.radio(
        "Date basis",
        ["Accounting date (date)"] + (["Bank reconciliation (reconcile)"] if has_reconcile else []),
        horizontal=True
    )
    if not has_reconcile:
        st.info("Bank reconciliation timeline not available (column 'reconcile' not found).")

    # ---- Filters ----
    bill_like = st.text_input("Filter bill_no contains", value="")
    if bill_like.strip():
        where_bill = "and coalesce(bill_no,'') ilike %(bill_like)s"
        params["bill_like"] = f"%{bill_like.strip()}%"
    else:
        where_bill = ""

    # ---- Sample ledger rows (lightweight, used for classification) ----
    cols = ["date", "head_name", "subhead1", "account", "bill_no", "folio_chq_no", "bank", "debit_payment", "credit_deposit"]
    col_sql = ", ".join([f'"{c}"' if c == "date" else c for c in cols])
    if has_reconcile:
        col_sql += ", reconcile"

    sql_sample = f"""
        select {col_sql}
        from {rel}
        where {where_sql}
        {where_bill}
        limit 5000
    """
    df0 = run_df(engine, sql_sample, params, rel=rel)
    if df0 is None or df0.empty:
        st.warning("No billing ledger rows found under current filters.")
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
    df0["is_indirect_income"] = (
        df0["subhead1"].str.contains("Income \(Indirect/Opr\.", case=False, na=False)
        | df0["subhead1"].str.contains("Income \(Indirect/Opr\)", case=False, na=False)
    )

    df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
    if has_reconcile:
        df0["reconcile"] = pd.to_datetime(df0["reconcile"], errors="coerce")

    # ---- Reconciliation key ----
    st.divider()
    st.markdown("## 🔗 Reconciliation key")

    suggested = _infer_key_mode_choice(df0)
    key_mode = st.selectbox(
        "How to reconcile (logical bill key)",
        ["auto", "bill_no only", "bill_no + pay_to", "bill_no + account", "account-ledger (account + subhead1)"],
        index=0
    )
    if key_mode == "auto":
        st.caption(f"Auto suggestion based on data: **{suggested}**")

    if key_mode == "auto":
        key_mode_effective = "account-ledger (account + subhead1)" if suggested == "account-ledger" else "bill_no + account"
    else:
        key_mode_effective = key_mode

    # key builder
    if key_mode_effective == "account-ledger (account + subhead1)":
        df0["bill_key"] = df0["account"].str.strip() + " | " + df0["subhead1"].str.strip()
    elif key_mode_effective == "bill_no only":
        df0["bill_key"] = df0["bill_no"].str.strip()
    elif key_mode_effective == "bill_no + account":
        df0["bill_key"] = df0["bill_no"].str.strip() + " | " + df0["account"].str.strip()
    elif key_mode_effective == "bill_no + pay_to":
        if "pay_to" in df0.columns:
            df0["bill_key"] = df0["bill_no"].str.strip() + " | " + df0["pay_to"].astype(str).str.strip()
        else:
            df0["bill_key"] = df0["bill_no"].str.strip()
            st.info("Column pay_to not found; fallback to bill_no only.")
    else:
        df0["bill_key"] = df0["bill_no"].str.strip()

    # ---- Grants separated from ordinary receipts ----
    st.divider()
    st.markdown("## 🏷 Income classification (Grants vs Ordinary)")

    receipts_total = float(df0.loc[df0["credit_deposit"] > 0, "credit_deposit"].sum())
    grant_receipts = float(df0.loc[(df0["credit_deposit"] > 0) & (df0["is_grant"]), "credit_deposit"].sum())
    ordinary_receipts = receipts_total - grant_receipts
    grant_ratio = (grant_receipts / receipts_total * 100.0) if receipts_total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Receipts total", _money(receipts_total))
    c2.metric("Grant receipts", _money(grant_receipts))
    c3.metric("Grant dependency %", f"{grant_ratio:.1f}%")
    st.caption("Government Grant is reported separately from ordinary income streams.")

    # ---- Indirect income rollups ----
    st.divider()
    st.markdown("## 🧩 Indirect/Opr income rollups (subhead1 → head_name → account)")

    df_ind = df0[(df0["is_indirect_income"]) & (df0["credit_deposit"] > 0)].copy()
    if df_ind.empty:
        st.info("No rows found under subhead1 = Income (Indirect/Opr.) within current filters.")
    else:
        roll1 = df_ind.groupby(["subhead1"], as_index=False).agg(
            receipts=("credit_deposit", "sum"),
            rows=("credit_deposit", "size"),
        ).sort_values("receipts", ascending=False)
        st.markdown("### By subhead1")
        show_df(roll1)

        roll2 = df_ind.groupby(["subhead1", "head_name"], as_index=False).agg(
            receipts=("credit_deposit", "sum"),
            rows=("credit_deposit", "size"),
        ).sort_values("receipts", ascending=False)
        st.markdown("### By subhead1 → head_name")
        show_df(roll2.head(200))

        roll3 = df_ind.groupby(["subhead1", "head_name", "account"], as_index=False).agg(
            receipts=("credit_deposit", "sum"),
            rows=("credit_deposit", "size"),
        ).sort_values("receipts", ascending=False)
        st.markdown("### By subhead1 → head_name → account")
        show_df(roll3.head(300))

    # ---- Ledger reconciliation ----
    st.divider()
    st.markdown("## 📚 Billing ledger reconciliation (issued vs received vs outstanding)")

    df_led = df0[(df0["debit_payment"] != 0) | (df0["credit_deposit"] != 0)].copy()

    # Force PAR/WAR/AMC/AGR routing to account-ledger key
    df_led["bill_key_routed"] = df_led["bill_key"]
    par_mask = df_led["billing_stream"].isin(PAR_STREAMS)
    df_led.loc[par_mask, "bill_key_routed"] = df_led.loc[par_mask, "account"].str.strip() + " | " + df_led.loc[par_mask, "subhead1"].str.strip()

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

    # Aging basis
    if ("Bank reconciliation" in date_basis) and has_reconcile:
        base_dt = pd.to_datetime(agg["last_reconcile"], errors="coerce")
    else:
        base_dt = pd.to_datetime(agg["last_activity"], errors="coerce")

    today = pd.to_datetime(date.today())
    agg["days_since"] = (today - base_dt).dt.days
    agg["aging_bucket"] = agg["days_since"].apply(_aging_bucket)

    st.markdown("### Top outstanding")
    show_df(agg.sort_values("outstanding", ascending=False).head(200))

    st.markdown("### Aging bucket totals (outstanding)")
    buckets = agg.groupby("aging_bucket", as_index=False).agg(
        items=("bill_key_routed", "count"),
        outstanding=("outstanding", "sum"),
    ).sort_values("aging_bucket")
    show_df(buckets)

    # ---- Control flags ----
    st.divider()
    st.markdown("## 🧾 Controls & exceptions")

    unapplied = df0[(df0["credit_deposit"] > 0) & (df0["bill_no"].str.strip() == "")].copy()
    st.markdown("### Receipts without bill_no reference")
    show_df(unapplied.sort_values("date", ascending=False).head(300))

    missing_receipt_no = df0[(df0["credit_deposit"] > 0) & (df0["folio_chq_no"].str.strip() == "")].copy()
    st.markdown("### Receipts missing receipt no (folio_chq_no)")
    show_df(missing_receipt_no.sort_values("date", ascending=False).head(300))

    # ---- Deposit routing ----
    st.divider()
    st.markdown("## 🏦 Deposit routing (bank)")

    by_bank = df0[df0["credit_deposit"] > 0].groupby(["bank"], as_index=False).agg(
        receipts=("credit_deposit", "sum"),
        rows=("credit_deposit", "size"),
        with_receipt_no=("folio_chq_no", lambda s: (s.astype(str).str.strip() != "").sum()),
    ).sort_values("receipts", ascending=False)
    show_df(by_bank.head(50))

    st.markdown("### Potential bank posting mismatch (same routed key across multiple banks)")
    tmp = df0[df0["credit_deposit"] > 0].copy()
    tmp["bill_key_routed"] = tmp["bill_key"]
    par_mask2 = tmp["billing_stream"].isin(PAR_STREAMS)
    tmp.loc[par_mask2, "bill_key_routed"] = tmp.loc[par_mask2, "account"].str.strip() + " | " + tmp.loc[par_mask2, "subhead1"].str.strip()

    bank_mix = tmp.groupby(["bill_key_routed"], as_index=False).agg(
        banks=("bank", lambda s: len(set([x for x in s.astype(str) if x.strip()]))),
        receipts=("credit_deposit", "sum"),
        last_date=("date", "max"),
    )
    bank_mix = bank_mix[bank_mix["banks"] >= 2].sort_values("receipts", ascending=False).head(200)
    show_df(bank_mix)
