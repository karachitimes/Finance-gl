import streamlit as st
import pandas as pd
import calendar
import re
from datetime import date, datetime
from difflib import get_close_matches
from sqlalchemy import text

from db import run_df, run_scalar, get_engine
from semantic import (
    RECoup_START_DATE,
    REL_SEM, REL_EXP, REL_CF, REL_REV,
    expense_amount_expr, cashflow_net_expr, cashflow_dir_expr,
    tb_amount_expr, has_column, pick_view,
)
from filters import build_where_from_ui, USE_UI
from utils import show_df, show_pivot

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

# -----------------------------
# Cached lists (DO NOT accept engine)
# -----------------------------
@st.cache_data(ttl=3600)
def get_known_payees_cached(rel: str):
    engine = get_engine()
    q = text(f"select distinct pay_to from {rel} where pay_to is not null order by 1")
    with engine.connect() as conn:
        return [str(r[0]) for r in conn.execute(q).fetchall()]

def get_known_payees(engine, rel: str):
    # keep signature; engine unused
    return get_known_payees_cached(rel)

@st.cache_data(ttl=3600)
def get_known_func_codes_cached(rel: str):
    engine = get_engine()
    q = text(f"select distinct func_code from {rel} where func_code is not null order by 1")
    with engine.connect() as conn:
        return [str(r[0]) for r in conn.execute(q).fetchall()]

def get_known_func_codes(engine, rel: str):
    return get_known_func_codes_cached(rel)


def best_payee_match(name: str | None, known_payees: list[str]):
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    matches = get_close_matches(name.title(), known_payees, n=1, cutoff=0.75)
    return matches[0] if matches else None


# -----------------------------
# Parsing helpers
# -----------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def not_recoup_filter(q: str) -> bool:
    t = _norm(q)
    return (
        ("not recoup" in t)
        or ("exclude recoup" in t)
        or ("without recoup" in t)
        or ("bill_no not recoup" in t)
    )


def parse_pay_to(q: str) -> str | None:
    """Detect patterns like 'pay to Ahmed', 'pay to "ABC Traders"', 'payto xyz'"""
    if not q:
        return None
    t = q.strip()

    # pay to "Name Here"
    m = re.search(r'pay\s*to\s*["\']([^"\']+)["\']', t, flags=re.I)
    if m:
        return m.group(1).strip()

    # pay to Name Here
    m = re.search(r'pay\s*to\s+([a-zA-Z0-9 ._&-]+)', t, flags=re.I)
    if m:
        return m.group(1).strip()

    # payto Name
    m = re.search(r'payto\s+([a-zA-Z0-9 ._&-]+)', t, flags=re.I)
    if m:
        return m.group(1).strip()

    return None
    
def wants_monthly(q: str) -> bool:
    t = _norm(q)
    return ("monthly" in t) or ("monthwise" in t) or ("month wise" in t) or ("by month" in t) or ("trend" in t)

def wants_by_head(q: str) -> bool:
    t = _norm(q)
    return ("by head" in t) or ("head wise" in t) or ("headwise" in t) or ("head-wise" in t)

def wants_by_bank(q: str) -> bool:
    t = _norm(q)
    return ("by bank" in t) or ("bank wise" in t) or ("bankwise" in t) or ("bank-wise" in t)

def parse_top_n(q: str) -> int | None:
    t = _norm(q)
    m = re.search(r"\btop\s*([0-9]{1,3})\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    if "top ten" in t:
        return 10
    return None

def parse_recoup_state(q: str) -> str | None:
    t = _norm(q)
    if "pending" in t:
        return "pending"
    if "completed" in t or "complete" in t:
        return "completed"
    return None

def parse_as_of_date(q: str) -> date | None:
    t = _norm(q)
    m = re.search(r"\bas of\s+([0-9]{4}-[0-9]{2}-[0-9]{2})\b", t)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()

    m = re.search(r"\bas of\s+([0-9]{1,2})[/-]([0-9]{1,2})[/-]([0-9]{4})\b", t)
    if m:
        dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(yyyy, mm, dd)
    return None

def parse_field_search(q: str) -> tuple[str | None, str | None]:
    m = re.search(r"\bsearch\s+(voucher_no|reference_no|bill_no|status|account|bank|head_name|pay_to)\s+(.+)$", q.strip(), flags=re.I)
    if not m:
        return None, None
    return m.group(1).lower(), m.group(2).strip()


def detect_intent(q: str) -> str:
    ql = q.lower().strip()

    # compliance / policy checks
    if any(p in ql for p in ["violation", "violations", "policy", "compliance", "without bill", "no bill reference", "head mapping", "unmapped head"]):
        return "compliance"


    # trial balance
    if "trial balance" in ql or "tb" in ql or "balance as of" in ql:
        return "trial_balance"

    if "cashflow" in ql or "cash flow" in ql:
        return "cashflow"

    if any(w in ql for w in ["expense", "cost", "paid", "payment", "wages", "salary"]):
        return "expense"

    if "recoup" in ql:
        return "recoup"

    if any(w in ql for w in ["revenue", "income", "grant"]):
        return "revenue"

    if ql.startswith("search "):
        return "search"

    return "search"


def detect_structure(q: str):
    return {
        "by_head": wants_by_head(q),
        "by_bank": wants_by_bank(q),
        "monthly": wants_monthly(q),
        "top": parse_top_n(q) is not None,
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
        return (
            """
            "date" >= date_trunc('month', current_date) - interval '1 month'
            and "date" <  date_trunc('month', current_date)
            """.strip(),
            {}
        )
    if "this month" in ql:
        return (
            """
            "date" >= date_trunc('month', current_date)
            and "date" <  date_trunc('month', current_date) + interval '1 month'
            """.strip(),
            {}
        )
    return None, {}


def extract_payee(q: str, known_payees: list[str]):
    ql = q.lower()
    m = re.search(r"(?:to)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        return best_payee_match(m.group(1), known_payees)
    return None


def apply_intent_func_override(intent: str, question: str):
    if intent == "revenue":
        return "Revenue"
    # expense/recoup/cashflow/tb/search should not force func_code unless user selected one in UI
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "search"):
        return None
    return USE_UI


def render_qa_tab(engine, f, *, rel: str):
    st.subheader("Ask a Finance Question (Deterministic + Search)")
    st.caption("Examples: revenue by head monthly | expense by head monthly | expense by head top 10 | cashflow by bank monthly | recoup pending | trial balance as of 2026-01-31 | search voucher_no 1234")

    q = st.text_input("Ask anything…", placeholder="expense by head monthly")
    if not q:
        return

    # Prefer semantic views if present
    rel0 = pick_view(engine, REL_SEM, rel)

    known_payees = get_known_payees(engine, rel0)
    intent = detect_intent(q)
    struct = detect_structure(q)
    payee = extract_payee(q, known_payees)
    func_override = apply_intent_func_override(intent, q)

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=func_override
    )
    # pay_to filter from question text (deterministic)
    pay_to_name = parse_pay_to(q)
    if pay_to_name:
        where.append("pay_to ilike :pay_to_name")
        params["pay_to_name"] = f"%{pay_to_name}%"

    # "as of" date for trial balance
    asof = parse_as_of_date(q)
    if asof and intent == "trial_balance":
        params["dt"] = asof  # keep df same; just cap at asof

    # quick date phrases
    date_sql, date_params = infer_date_sql(q)
    if date_sql:
        where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
        where.insert(0, date_sql)
        params.update(date_params)

    # month range like "Jan to Mar"
    m_start, m_end_excl = parse_month_range(q)
    if m_start and m_end_excl:
        where = [w for w in where if "between :df and :dt" not in w and '"date" between' not in w]
        where.insert(0, '"date" >= :m_start and "date" < :m_end')
        params["m_start"] = m_start
        params["m_end"] = m_end_excl

    if payee:
        where.append("pay_to ilike :payee")
        params["payee"] = f"%{payee}%"

    # expense modifier: not recoup
    if intent == "expense" and not_recoup_filter(q):
        where.append("coalesce(bill_no,'') <> 'Recoup'")

    # expense modifier: folio cheque blank (optional)
    if intent == "expense":
        t = (q or "").lower()
        if ("folio_chq_no" in t or "folio chq no" in t or "folio cheque" in t or "folio cheq" in t) and ("blank" in t or "empty" in t or "null" in t):
            if has_column(engine, REL_EXP, "folio_chq_no"):

                where.append("NULLIF(TRIM(COALESCE(folio_chq_no,'')),'') IS NULL")
            else:
                st.warning("folio_chq_no column not available in expense view for filtering.")

        
        # pay_to filter from question text
    pay_to_name = parse_pay_to(q)
    if pay_to_name and has_column(REL_SEM, "pay_to"):
        where.append("pay_to ilike :pay_to_name")
        params["pay_to_name"] = f"%{pay_to_name}%"
    
    where_sql = " and ".join(where) if where else "1=1"

    # -----------------------------
    # Revenue
    # -----------------------------
    if intent == "revenue":
        # Use revenue view if available, else rel0
        relr = pick_view(engine, REL_REV, rel0)

        if struct["by_head"] and struct["monthly"]:
            st.subheader("Revenue by Head (Monthly) — Pivot")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   head_name,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {relr}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1,2
            order by 1,3 desc
            """
            df_out = run_df(engine, sql, params, ["Month","Head","Revenue"], rel=relr)
            if df_out.empty:
                st.info("No rows for selected filters.")
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
            from {relr}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1
            order by 2 desc
            limit 50
            """
            show_df(run_df(engine, sql, params, ["Head","Revenue"], rel=relr), label_col="Head")
            return

        if struct["monthly"]:
            st.subheader("Monthly Revenue")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   to_char(date_trunc('month',"date"),'Mon-YY') as month_label,
                   sum(coalesce(credit_deposit,0)) as revenue
            from {relr}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
            group by 1,2
            order by 1
            """
            show_df(run_df(engine, sql, params, ["Month","Month Label","Revenue"], rel=relr), label_col="Month Label")
            return

        val = run_scalar(engine, f"""
            select coalesce(sum(coalesce(credit_deposit,0)),0)
            from {relr}
            where {where_sql}
              and func_code = 'Revenue'
              and coalesce(credit_deposit,0) > 0
        """, params, rel=relr)
        st.success(f"Total Revenue: {val:,.0f} PKR")
        return
 # ---- Compliance / Policy ----
    if intent == "compliance":
        st.subheader("Compliance / Policy Checks")

        ql = q.lower()

        # 1) Recoup policy violations (practical definition):
        #    - bill_no='Recoup' AND status blank AND older than N days
        if "recoup" in ql and ("violation" in ql or "policy" in ql or "compliance" in ql):
            # Default: flag pending recoup older than 30 days
            days = 30
            m = re.search(r"\b(\d{1,3})\s*days\b", ql)
            if m:
                days = int(m.group(1))

            sql = f"""
                select
                    "date",
                    bank,
                    account,
                    head_name,
                    pay_to,
                    bill_no,
                    status,
                    voucher_no,
                    reference_no,
                    (coalesce(debit_payment,0) - coalesce(credit_deposit,0)) as amount,
                    (current_date - "date") as age_days
                from {rel0}
                where {where_sql}
                  and bill_no = 'Recoup'
                  and nullif(trim(coalesce(status,'')),'') is null
                  and (current_date - "date") >= :days
                order by "date" asc
                limit 500
            """
            params2 = dict(params)
            params2["days"] = days
            df_out = run_df(engine, sql, params2, rel=rel0)
            show_df(df_out)
            return

        # 2) Payments without bill reference
        if ("without bill reference" in ql) or ("no bill reference" in ql) or ("bill reference" in ql and "without" in ql):
            sql = f"""
                select
                    "date", bank, account, head_name, pay_to,
                    voucher_no, reference_no, bill_no, status,
                    coalesce(debit_payment,0) as debit_payment,
                    coalesce(credit_deposit,0) as credit_deposit,
                    description
                from {rel0}
                where {where_sql}
                  and coalesce(debit_payment,0) > 0
                  and nullif(trim(coalesce(bill_no,'')),'') is null
                order by "date" desc
                limit 500
            """
            df_out = run_df(engine, sql, params, rel=rel0)
            show_df(df_out)
            return

        # 3) Expenses without approved head mapping
        # Option A (no mapping table): treat blank head_name as unmapped
        if ("head mapping" in ql) or ("unmapped head" in ql) or ("without approved head" in ql):
            rele = pick_view(engine, REL_EXP, rel0)
            exp_expr = expense_amount_expr(engine, rele)
            sql = f"""
                select
                    "date", bank, account, pay_to,
                    head_name, voucher_no, reference_no,
                    {exp_expr} as amount,
                    description
                from {rele}
                where {where_sql}
                  and (nullif(trim(coalesce(head_name,'')),'') is null)
                order by "date" desc
                limit 500
            """
            df_out = run_df(engine, sql, params, rel=rele)
            show_df(df_out)
            return

        st.info("Try: 'recoup policy violations 60 days' | 'payments without bill reference' | 'expenses without approved head mapping'")
        return
    # -----------------------------
    # Expense
    # -----------------------------
    if intent == "expense":
        # Use expense view if available, else rel0
        rele = pick_view(engine, REL_EXP, rel0)
        exp_expr = expense_amount_expr(engine, rele)

        top_n = parse_top_n(q)
        by_head = struct["by_head"]
        monthly = struct["monthly"]

        if by_head and monthly:
            st.subheader("Expense by Head (Monthly) — Pivot")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   head_name,
                   sum({exp_expr}) as outflow
            from {rele}
            where {where_sql}
            group by 1,2
            order by 1,3 desc
            """
            df_out = run_df(engine, sql, params, ["Month","Head","Outflow"], rel=rele)
            if df_out.empty:
                st.info("No rows for selected filters.")
                return
            df_out["Month"] = pd.to_datetime(df_out["Month"])
            pv = df_out.pivot_table(index="Head", columns="Month", values="Outflow", aggfunc="sum", fill_value=0)
            pv = pv.reindex(sorted(pv.columns), axis=1)
            pv.columns = [d.strftime("%b-%y") for d in pv.columns]
            show_pivot(pv)
            return

        if by_head:
            label = "Expense by Head"
            if top_n:
                label += f" (Top {top_n})"
            st.subheader(label)

            limit_sql = f"limit {int(top_n)}" if top_n else "limit 50"
            sql = f"""
            select head_name, sum({exp_expr}) as outflow
            from {rele}
            where {where_sql}
            group by 1
            order by 2 desc
            {limit_sql}
            """
            show_df(run_df(engine, sql, params, ["Head","Outflow"], rel=rele), label_col="Head")
            return

        if monthly:
            st.subheader("Monthly Expense")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   to_char(date_trunc('month',"date"),'Mon-YY') as month_label,
                   sum({exp_expr}) as outflow
            from {rele}
            where {where_sql}
            group by 1,2
            order by 1
            """
            show_df(run_df(engine, sql, params, ["Month","Month Label","Outflow"], rel=rele), label_col="Month Label")
            return

        val = run_scalar(engine, f"""
            select coalesce(sum({exp_expr}),0)
            from {rele}
            where {where_sql}
        """, params, rel=rele)
        st.success(f"Total Expense Outflow: {val:,.0f} PKR")
        return

    # -----------------------------
    # Cashflow
    # -----------------------------
    if intent == "cashflow":
        relc = pick_view(engine, REL_CF, rel0)
        net_expr = cashflow_net_expr(engine, relc)
        dir_expr = cashflow_dir_expr(engine, relc, net_expr)

        if struct["monthly"]:
            st.subheader("Cashflow by Bank (Monthly) — Pivot")
            sql = f"""
            select date_trunc('month',"date")::date as month,
                   coalesce(bank,'UNKNOWN') as bank,
                   sum({net_expr}) as amount
            from {relc}
            where {where_sql}
            group by 1,2
            order by 1,2
            """
            df_out = run_df(engine, sql, params, ["Month","Bank","Amount"], rel=relc)
            if df_out.empty:
                st.info("No rows for selected filters.")
                return
            df_out["Month"] = pd.to_datetime(df_out["Month"])
            pv = df_out.pivot_table(index="Bank", columns="Month", values="Amount", aggfunc="sum", fill_value=0)
            pv = pv.reindex(sorted(pv.columns), axis=1)
            pv.columns = [d.strftime("%b-%y") for d in pv.columns]
            show_pivot(pv)
            return

        st.subheader("Cashflow by Bank & Direction")
        sql = f"""
            select coalesce(bank,'UNKNOWN') as bank,
                   {dir_expr} as direction,
                   sum({net_expr}) as amount
            from {relc}
            where {where_sql}
            group by 1,2
            order by 1,2
        """
        show_df(run_df(engine, sql, params, ["Bank","Direction","Amount"], rel=relc), label_col="Bank")
        return

    # -----------------------------
    # Trial Balance
    # -----------------------------
    if intent == "trial_balance":
        st.subheader("Trial Balance" + (f" (As of {params.get('dt')})" if asof else ""))
        relt = rel0
        amt_expr = tb_amount_expr(engine, relt)
        sql = f"""
            select account, sum({amt_expr}) as balance
            from {relt}
            where {where_sql}
            group by 1
            order by 1
        """
        show_df(run_df(engine, sql, params, ["Account","Balance"], rel=relt), label_col="Account")
        return

    # -----------------------------
    # Recoup
    # -----------------------------
    if intent == "recoup":
        st.subheader("Recoup")
        state = parse_recoup_state(q)

        conds = ["bill_no = 'Recoup'"]
        # If your semantic view doesn't enforce bank scoping for recoup, uncomment:
        # conds.append("coalesce(bank,'') = 'Revenue:4069284635'")

        if state == "pending":
            st.caption("Showing: Recoup Pending")
            conds.append("nullif(trim(coalesce(status,'')),'') is null")
        elif state == "completed":
            st.caption("Showing: Recoup Completed")
            conds.append("nullif(trim(coalesce(status,'')),'') is not null")

        sql = f"""
        select
          case
            when nullif(trim(coalesce(status,'')),'') is null then 'pending'
            else 'completed'
          end as recoup_state,
          coalesce(sum(coalesce(debit_payment,0) - coalesce(credit_deposit,0)),0) as amount
        from {rel0}
        where {where_sql}
          and {" and ".join(conds)}
        group by 1
        order by 1
        """
        df_out = run_df(engine, sql, params, ["State","Amount"], rel=rel0)
        show_df(df_out, label_col="State")
        return

    # -----------------------------
    # Search (field-aware)
    # -----------------------------
    st.subheader("Search (latest rows)")

    field, value = parse_field_search(q)
    params2 = dict(params)

    if field and value:
        params2["q"] = f"%{value}%"
        if has_column(engine, rel0, field):
            field_sql = f"coalesce({field},'') ilike :q"
        else:
            field_sql = "coalesce(description,'') ilike :q"
    else:
        term = q.strip()
        params2["q"] = f"%{term}%"
        field_sql = """
          coalesce(description,'') ilike :q
          or coalesce(pay_to,'') ilike :q
          or coalesce(account,'') ilike :q
          or coalesce(head_name,'') ilike :q
          or coalesce(bill_no,'') ilike :q
        """

    base_cols = ['"date"', "bank", "account", "head_name", "pay_to", "description"]
    optional_cols = ["debit_payment", "credit_deposit", "gl_amount", "net_flow", "bill_no", "status", "voucher_no", "reference_no", "func_code", "attribute"]
    select_cols = []
    for c in base_cols:
        if c == '"date"' or has_column(engine, rel0, c):
            select_cols.append(c)
    for c in optional_cols:
        if has_column(engine, rel0, c):
            select_cols.append(c)

    sql = f"""
        select {", ".join(select_cols)}
        from {rel0}
        where {where_sql}
          and ({field_sql})
        order by "date" desc
        limit 500
    """
    show_df(run_df(engine, sql, params2, rel=rel0))
