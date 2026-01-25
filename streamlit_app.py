import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from datetime import date
from difflib import get_close_matches
import calendar
import re

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Finance Analytics System", layout="wide")

# -------------------------------------------------
# DB / ENGINE
# -------------------------------------------------
@st.cache_resource
def get_engine():
    """Create a SQLAlchemy Engine with sensible defaults for Streamlit Cloud."""
    url = st.secrets["DATABASE_URL"]
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=3,
        max_overflow=2,
    )

def test_connection(engine):
    with engine.connect() as conn:
        return conn.execute(text("select 1")).scalar()

engine = get_engine()

# -------------------------------------------------
# UI HEADER + DB STATUS
# -------------------------------------------------
st.title("📊 Finance Analytics System")

try:
    _ = test_connection(engine)
    st.success("Database connected ✅")
except OperationalError as e:
    st.error("Database connection failed")
    try:
        real_err = str(e.orig)
    except Exception:
        real_err = str(e)
    st.code(real_err)
    st.stop()

# -------------------------------------------------
# HELPERS (DB)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def get_distinct(col: str):
    # col is controlled by code (not user), safe to interpolate.
    q = text(f'SELECT DISTINCT {col} FROM public.v_finance_logic WHERE {col} IS NOT NULL ORDER BY {col}')
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=3600)
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()
KNOWN_FUNC_CODES = get_distinct("func_code")

def best_payee_match(name: str | None):
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    matches = get_close_matches(name.title(), KNOWN_PAYEES, n=1, cutoff=0.75)
    return matches[0] if matches else name.title()

# -------------------------------------------------
# NLP-ish ROUTING (DETERMINISTIC)
# -------------------------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}

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

def parse_month_range(q: str, default_year: int | None = None):
    """Parses phrases like 'July to January' and returns (start_date, end_exclusive)."""
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
    """Returns (sql_fragment, params_dict) or (None, {})."""
    ql = q.lower()

    if "last month" in ql:
        return (
            "\"date\" >= date_trunc('month', current_date) - interval '1 month' "
            "and \"date\" < date_trunc('month', current_date)",
            {}
        )

    if "this month" in ql:
        return (
            "\"date\" >= date_trunc('month', current_date) "
            "and \"date\" < date_trunc('month', current_date) + interval '1 month'",
            {}
        )

    return None, {}



def extract_payee(q: str):
    ql = q.lower()
    m = re.search(r"(?:to|by)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        return best_payee_match(m.group(1))
    return None

def parse_explicit_func_code(q: str) -> str | None:
    """If user explicitly writes 'function X' / 'func_code X', return best match."""
    ql = q.lower()
    m = re.search(r"\b(?:function|func_code|func)\s*[:=]?\s*([a-z0-9_\-\s]+)", ql)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None
    matches = get_close_matches(raw.title(), KNOWN_FUNC_CODES, n=1, cutoff=0.6)
    return matches[0] if matches else raw.title()

def detect_structure(q: str):
    ql = q.lower()
    return {
        "by_head": ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql) or ("head" in ql and "by" in ql),
        "by_bank": ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql) or ("bank" in ql and "by" in ql),
        "monthly": ("monthly" in ql) or ("per month" in ql) or ("month wise" in ql) or ("month-wise" in ql) or ("trend" in ql),
        "top": ("top" in ql) or ("highest" in ql) or ("largest" in ql),
    }

# -------------------------------------------------
# FILTER BUILDER (keeps UI filters; override ONLY affects SQL)
# -------------------------------------------------
USE_UI = object()  # sentinel

def build_where_from_ui(df, dt, bank, head, account, func_code, *, func_override=USE_UI):
    """Build WHERE + params from UI. func_override affects ONLY func_code filter."""
    where = ['"date" between :df and :dt']
    params = {"df": df, "dt": dt}

    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head"); params["head"] = head
    if account != "ALL":
        where.append("account = :account"); params["account"] = account

    # Decide effective func filter
    if func_override is USE_UI:
        effective_func = func_code
        if effective_func == "ALL":
            effective_func = None
    elif func_override in (None, "ALL"):
        effective_func = None  # ignore func_code filter
    else:
        effective_func = func_override  # forced string

    if effective_func is not None:
        where.append("func_code = :func_code"); params["func_code"] = effective_func

    return where, params, (effective_func if effective_func is not None else "ALL")

def apply_intent_func_override(intent: str, question: str, ui_func_code: str):
    """Return USE_UI, None, or a string func_code."""
    explicit = parse_explicit_func_code(question)
    if explicit:
        return explicit  # respect explicit request

    if intent == "revenue":
        return "Revenue"  # force Revenue for revenue intent
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "search"):
        return None  # ignore UI func_code filter
    return USE_UI

# -------------------------------------------------
# UI FILTERS
# -------------------------------------------------
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        df = st.date_input("From Date", value=date(2025, 1, 1))
    with c2:
        dt = st.date_input("To Date", value=date.today())

    bank = st.selectbox("Bank", ["ALL"] + get_distinct("bank"))
    head = st.selectbox("Head", ["ALL"] + get_distinct("head_name"))
    account = st.selectbox("Account", ["ALL"] + get_distinct("account"))
    func_code = st.selectbox("Function Code", ["ALL"] + get_distinct("func_code"))

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_rev, tab_exp, tab_cf, tab_tb, tab_qa = st.tabs(
    ["Revenue", "Expense", "Cashflow", "Trial Balance", "AI Q&A"]
)

# ---------------- Revenue tab ----------------
with tab_rev:
    st.subheader("Revenue (Monthly)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, func_code, func_override="Revenue")

    sql = f"""
    select date_trunc('month', "date") as month,
           sum(coalesce(revenue_amount,0)) as revenue
    from public.v_finance_logic
    where {' and '.join(where)}
      and entry_type = 'revenue'
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_rev = pd.DataFrame(rows, columns=["Month", "Revenue"])
        st.dataframe(df_rev, use_container_width=True)
        st.line_chart(df_rev.set_index("Month"))
        st.success(f"Total Revenue: {df_rev['Revenue'].sum():,.0f} PKR")
    else:
        st.info("No revenue rows found for selected filters/date range.")

# ---------------- Expense tab ----------------
with tab_exp:
    st.subheader("Expenses (Monthly)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, func_code, func_override=None)

    sql = f"""
    select date_trunc('month', "date") as month,
           sum(coalesce(expense_amount,0)) as expense
    from public.v_finance_logic
    where {' and '.join(where)}
      and entry_type = 'expense'
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_exp = pd.DataFrame(rows, columns=["Month", "Expense"])
        st.dataframe(df_exp, use_container_width=True)
        st.line_chart(df_exp.set_index("Month"))
        st.success(f"Total Expense: {df_exp['Expense'].sum():,.0f} PKR")
    else:
        st.info("No expense rows found for selected filters/date range.")

# ---------------- Cashflow tab ----------------
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")
    where, params, _ = build_where_from_ui(df, dt, bank, head, account, func_code, func_override=None)

    sql = f"""
    select
      coalesce(bank, 'UNKNOWN') as bank,
      direction,
      sum(signed_amount) as amount
    from public.v_finance_logic
    where {' and '.join(where)}
    group by 1,2
    order by 1,2
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_cf = pd.DataFrame(rows, columns=["Bank", "Direction", "Amount"])
        st.dataframe(df_cf, use_container_width=True)

        inflow = df_cf[df_cf["Direction"] == "in"]["Amount"].sum()
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].sum()  # likely negative
        st.success(
            f"Inflow: {inflow:,.0f} PKR  |  Outflow: {abs(outflow):,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR"
        )
    else:
        st.info("No rows found for selected filters/date range.")

# ---------------- Trial balance tab ----------------
with tab_tb:
    st.subheader("Trial Balance (As of To Date)")

    where = ['"date" <= :dt']
    params = {"dt": dt}

    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head"); params["head"] = head
    if account != "ALL":
        where.append("account = :account"); params["account"] = account
    if func_code != "ALL":
        where.append("func_code = :func_code"); params["func_code"] = func_code

    sql = f"""
    select
      account,
      sum(signed_amount) as balance
    from public.v_finance_logic
    where {' and '.join(where)}
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    if rows:
        df_tb = pd.DataFrame(rows, columns=["Account", "Balance"])
        st.dataframe(df_tb, use_container_width=True)
        st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
    else:
        st.info("No rows found for trial balance with current filters.")

# ---------------- AI Q&A tab ----------------
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic + Search)")
    st.caption(
        "Examples: revenue by head | revenue by head monthly | expense by head | monthly revenue trend | pending recoup amount | trial balance"
    )

    q = st.text_input("Ask anything…", placeholder="revenue by head monthly")

    if q:
        intent = detect_intent(q)
        payee = extract_payee(q)

        # Build WHERE from UI + intent override (SQL-only)
        func_override = apply_intent_func_override(intent, q, func_code)
        where, params, effective_func = build_where_from_ui(df, dt, bank, head, account, func_code, func_override=func_override)

        # Override date filter if question specifies relative dates
        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            
            where = [
                w for w in where
                if "between :df and :dt" not in w
                and '"date" between' not in w
            ]
            where.insert(0, date_sql)
            params.update(date_params)

        # Month-range overrides date filter
        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [
                w for w in where
                if "between :df and :dt" not in w
                and '"date" between' not in w
                and '"date" >=' not in w
                and '"date" <' not in w
            ]

            where.insert(0, '"date" >= :m_start and "date" < :m_end')
            params["m_start"] = m_start
            params["m_end"] = m_end_excl

        if payee:
            where.append("pay_to ilike :payee")
            params["payee"] = f"%{payee}%"

        where_sql = " and ".join(where)
        struct = detect_structure(q)
        ql = q.lower()

        label = ""
        sql = ""

        # ---------- Revenue ----------
        if intent == "revenue":
            if struct["by_head"] and struct["monthly"]:
                label = "Revenue by Head (Monthly)"
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Revenue by Head"
                sql = f"""
                select head_name,
                       sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["by_bank"]:
                label = "Revenue by Bank"
                sql = f"""
                select coalesce(bank,'UNKNOWN') as bank,
                       sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 2 desc
                """
            elif struct["monthly"]:
                label = "Monthly Revenue"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 1
                """
            else:
                label = "Total Revenue"
                sql = f"""
                select coalesce(sum(coalesce(revenue_amount,0)),0) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                """

        # ---------- Expense ----------
        elif intent == "expense":
            if struct["by_head"] and struct["monthly"]:
                label = "Expense by Head (Monthly)"
                sql = f"""
                select date_trunc('month',"date") as month,
                       head_name,
                       sum(coalesce(expense_amount,0)) as expense
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='expense'
                group by 1,2
                order by 1,3 desc
                """
            elif struct["by_head"]:
                label = "Expense by Head"
                sql = f"""
                select head_name,
                       sum(coalesce(expense_amount,0)) as expense
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='expense'
                group by 1
                order by 2 desc
                limit 50
                """
            elif struct["monthly"]:
                label = "Monthly Expense"
                sql = f"""
                select date_trunc('month',"date") as month,
                       sum(coalesce(expense_amount,0)) as expense
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='expense'
                group by 1
                order by 1
                """
            else:
                label = "Total Expense"
                sql = f"""
                select coalesce(sum(coalesce(expense_amount,0)),0) as expense
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='expense'
                """

        # ---------- Recoup ----------
        elif intent == "recoup":
            pending = ("pending" in ql) or ("outstanding" in ql) or ("not recouped" in ql)
            recouped = ("recouped" in ql) or ("settled" in ql)

            if pending:
                label = "Pending Recoup Amount"
                sql = f"""
                select coalesce(sum(coalesce(recoup_pending_amount,0)),0) as pending_recoup
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='pending'
                """
            elif recouped:
                label = "Recouped Total"
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0) as recouped_total
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='recouped'
                """
            else:
                label = "Recoup Total"
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0) as recoup_total
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                """

        # ---------- Cashflow ----------
        elif intent == "cashflow":
            label = "Cashflow"
            sql = f"""
            select coalesce(bank,'UNKNOWN') as bank,
                   direction,
                   sum(signed_amount) as amount
            from public.v_finance_logic
            where {where_sql}
            group by 1,2
            order by 1,2
            """

        # ---------- Trial balance ----------
        elif intent == "trial_balance":
            label = "Trial Balance"
            sql = f"""
            select account,
                   sum(signed_amount) as balance
            from public.v_finance_logic
            where {where_sql}
            group by 1
            order by 1
            """

        # ---------- Search fallback ----------
        else:
            label = "Search matched total"
            params["q"] = q
            sql = f"""
            select coalesce(sum(signed_amount),0) as total
            from public.v_finance_logic
            where {where_sql}
              and (
                search_text % :q
                or search_tsv @@ plainto_tsquery('simple', :q)
              )
            """

        # ---------- Execute + Render ----------
        with engine.connect() as conn:
            if "group by" in sql.lower() or intent in ("cashflow", "trial_balance"):
                rows = conn.execute(text(sql), params).fetchall()
                if not rows:
                    st.warning("No rows found for this question with current filters.")
                else:
                    df_out = pd.DataFrame(rows)

                    # Set friendly column names when possible
                    if intent == "cashflow":
                        df_out.columns = ["Bank", "Direction", "Amount"]
                        st.dataframe(df_out, use_container_width=True)
                        inflow = df_out[df_out["Direction"] == "in"]["Amount"].sum()
                        outflow = df_out[df_out["Direction"] == "out"]["Amount"].sum()
                        st.success(f"Inflow: {inflow:,.0f} PKR | Outflow: {abs(outflow):,.0f} PKR | Net: {(inflow+outflow):,.0f} PKR")
                    elif intent == "trial_balance":
                        df_out.columns = ["Account", "Balance"]
                        st.dataframe(df_out, use_container_width=True)
                        st.success(f"Net (sum of balances): {df_out['Balance'].sum():,.0f} PKR")
                    else:
                        # heuristic naming
                        if "Revenue by Head (Monthly)" in label:
                            df_out.columns = ["Month", "Head", "Revenue"]
                        elif "Expense by Head (Monthly)" in label:
                            df_out.columns = ["Month", "Head", "Expense"]
                        elif label == "Monthly Revenue":
                            df_out.columns = ["Month", "Revenue"]
                            st.line_chart(df_out.set_index("Month"))
                        elif label == "Monthly Expense":
                            df_out.columns = ["Month", "Expense"]
                            st.line_chart(df_out.set_index("Month"))
                        elif label in ("Revenue by Head", "Expense by Head"):
                            df_out.columns = ["Head", "Amount"]
                        elif label == "Revenue by Bank":
                            df_out.columns = ["Bank", "Revenue"]

                        st.subheader(label)
                        st.dataframe(df_out, use_container_width=True)
            else:
                val = conn.execute(text(sql), params).scalar() or 0
                st.success(f"{label}: {val:,.0f} PKR")

        with st.expander("🔍 Why this result?"):
            st.write(f"Intent detected: `{intent}`")
            st.write(f"Effective function filter: `{effective_func}`")
            if payee:
                st.write(f"Payee: `{payee}`")
            if m_start and m_end_excl:
                st.write(f"Month range: `{m_start}` to `{m_end_excl}` (end exclusive)")
            st.write("Filters applied:")
            st.write(f"- Bank: `{bank}`  |  Head: `{head}`  |  Account: `{account}`  |  Function: `{func_code}`")
            st.write(f"- From: `{df}`  |  To: `{dt}`")
            st.write("SQL (debug):")
            st.code(sql.strip())
            st.write("Params (debug):")
            st.json({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in params.items()})
