import streamlit as st
import pandas as pd
import calendar
import re
from datetime import date
from difflib import get_close_matches

from db import run_df, run_scalar
from semantic import RECoup_START_DATE, get_source_relation
from semantic import has_column, expense_amount_expr, cashflow_net_expr, cashflow_dir_expr, tb_amount_expr
from filters import build_where_from_ui, USE_UI
from utils import show_df, show_pivot

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

@st.cache_data(ttl=3600)
def get_known_payees(engine, rel: str):
    from filters import get_distinct
    return get_distinct(engine, rel, "pay_to")

@st.cache_data(ttl=3600)
def get_known_func_codes(engine, rel: str):
    from filters import get_distinct
    return get_distinct(engine, rel, "func_code")

def best_payee_match(name: str | None, known_payees: list[str]):
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    matches = get_close_matches(name.title(), known_payees, n=1, cutoff=0.75)
    return matches[0] if matches else None

def detect_intent(q: str) -> str:
    ql = q.lower()
    if "trial balance" in ql or "tb" in ql or "balance as of" in ql:
        return "trial_balance"
    if "cashflow" in ql or "cash flow" in ql:
        return "cashflow"
    if "recoup" in ql:
        return "recoup"
    if any(w in ql for w in ["revenue", "income", "grant"]):
        return "revenue"
    if any(w in ql for w in ["expense", "cost", "paid", "payment", "wages", "salary"]):
        return "expense"
    return "search"

def detect_structure(q: str):
    ql = q.lower()
    return {
        "by_head": ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql) or ("head" in ql and "by" in ql),
        "by_bank": ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql) or ("bank" in ql and "by" in ql),
        "monthly": ("monthly" in ql) or ("per month" in ql) or ("month wise" in ql) or ("month-wise" in ql) or ("trend" in ql),
        "top": ("top" in ql) or ("highest" in ql) or ("largest" in ql),
    }

def parse_month_range(q: str, default_year: int | None = None):
    ql = q.lower()
    default_year = default_year or date.today().year
    month_pattern = r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
    found = re.findall(month_pattern, ql)
    found = [m.lower() for m in found if m.lower() in MONTHS]
    if len(found) >= 2:
        m1, m2 = MONTHS[found[0]], MONTHS[found[1]]
        y1 = default_year
        y2 = default_year + (1 if m2 < m1 else 0)
        start = date(y1, m1, 1)
        end_excl = date(y2 + 1, 1, 1) if m2 == 12 else date(y2, m2 + 1, 1)
        return start, end_excl
    return None, None

def infer_date_sql(q: str):
    ql = q.lower()
    if "last month" in ql:
        return (""""date" >= date_trunc('month', current_date) - interval '1 month'
and "date" <  date_trunc('month', current_date)""", {})
    if "this month" in ql:
        return (""""date" >= date_trunc('month', current_date)
and "date" <  date_trunc('month', current_date) + interval '1 month'""", {})
    return None, {}

def extract_payee(q: str, known_payees: list[str]):
    ql = q.lower()
    m = re.search(r"(?:to)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        return best_payee_match(m.group(1), known_payees)
    return None

def apply_intent_func_override(intent: str, question: str):
    # keep it deterministic and compatible with your existing logic
    if intent == "revenue":
        return "Revenue"
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "search"):
        return None
    return USE_UI

def _is_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NULL"

def _not_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NOT NULL"

def render_qa_tab(engine, f, *, rel: str):
    st.subheader("Ask a Finance Question (Deterministic + Search)")
    st.caption("Examples: revenue by head monthly | expense by head | cashflow by bank | pending recoup | trial balance | search vendor name")

    q = st.text_input("Ask anything…", placeholder="revenue by head monthly")
    if not q:
        return

    rel0 = rel
    known_payees = get_known_payees(engine, rel0)
    intent = detect_intent(q)
    struct = detect_structure(q)
    payee = extract_payee(q, known_payees)
    func_override = apply_intent_func_override(intent, q)

    where, params, effective_func = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=func_override
    )

    date_sql, date_params = infer_date_sql(q)
    if date_sql:
        where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
        where.insert(0, date_sql.strip())
        params.update(date_params)

    m_start, m_end_excl = parse_month_range(q)
    if m_start and m_end_excl:
        where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
        where.insert(0, '"date" >= :m_start and "date" < :m_end')
        params["m_start"] = m_start
        params["m_end"] = m_end_excl

    if payee:
        where.append("pay_to ilike :payee")
        params["payee"] = f"%{payee}%"

    where_sql = " and ".join(where) if where else "1=1"

    # Routing
    if intent == "revenue":
        if struct["by_head"] and struct["monthly"]:
            st.subheader("Revenue by Head (Monthly) — Pivot")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   head_name,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {rel0}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1,2
            order by 1,3 desc
            """
            df_out = run_df(engine, sql, params, ["Month","Head","Revenue"], rel=rel0)
            if df_out.empty:
                st.warning("No rows found.")
                return
            df_out["Month"] = pd.to_datetime(df_out["Month"])
            pv = df_out.pivot_table(index="Head", columns="Month", values="Revenue", aggfunc="sum", fill_value=0)
            pv = pv.reindex(sorted(pv.columns), axis=1)
            pv.columns = [d.strftime("%b-%y") for d in pv.columns]
            show_pivot(pv)
            return

        if struct["by_head"]:
            st.subheader("Revenue by Head")
            sql = f"""
            select head_name,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {rel0}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1
            order by 2 desc
            limit 50
            """
            show_df(run_df(engine, sql, params, ["Head","Revenue"], rel=rel0), label_col="Head")
            return

        if struct["monthly"]:
            st.subheader("Monthly Revenue")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   to_char(date_trunc('month',"date"),'Mon-YY') as month_label,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {rel0}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1,2
            order by 1
            """
            show_df(run_df(engine, sql, params, ["Month","Month Label","Revenue"], rel=rel0), label_col="Month Label")
            return

        val = run_scalar(engine, f"""select coalesce(sum(coalesce(credit_deposit,0)),0) from {rel0} where {where_sql}
            and func_code = 'Revenue' and coalesce(credit_deposit,0) > 0""", params, rel=rel0)
        st.success(f"Total Revenue: {val:,.0f} PKR")
        return

    if intent == "expense":
        st.subheader("Expense")
        exp_expr = expense_amount_expr(engine, rel0)
        sql = f"""select head_name, sum({exp_expr}) as outflow
                  from {rel0}
                  where {where_sql}
                  group by 1
                  order by 2 desc
                  limit 50"""
        show_df(run_df(engine, sql, params, ["Head","Outflow"], rel=rel0), label_col="Head")
        return

    if intent == "cashflow":
        st.subheader("Cashflow by Bank & Direction")
        net_expr = cashflow_net_expr(engine, rel0)
        dir_expr = cashflow_dir_expr(engine, rel0, net_expr)
        sql = f"""select coalesce(bank,'UNKNOWN') as bank, {dir_expr} as direction, sum({net_expr}) as amount
                  from {rel0} where {where_sql}
                  group by 1,2 order by 1,2"""
        show_df(run_df(engine, sql, params, ["Bank","Direction","Amount"], rel=rel0), label_col="Bank")
        return

    if intent == "trial_balance":
        st.subheader("Trial Balance")
        amt_expr = tb_amount_expr(engine, rel0)
        sql = f"""select account, sum({amt_expr}) as balance from {rel0} where {where_sql} group by 1 order by 1"""
        show_df(run_df(engine, sql, params, ["Account","Balance"], rel=rel0), label_col="Account")
        return

    # search
    st.subheader("Search (latest rows)")
    params2 = dict(params)
    params2["q"] = f"%{q.strip()}%"
    sql = f"""select "date", bank, account, head_name, pay_to, description
             from {rel0}
             where {where_sql}
               and (coalesce(description,'') ilike :q or coalesce(pay_to,'') ilike :q or coalesce(account,'') ilike :q or coalesce(head_name,'') ilike :q)
             order by "date" desc
             limit 200"""
    show_df(run_df(engine, sql, params2, rel=rel0))
