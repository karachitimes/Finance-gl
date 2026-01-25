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
    """
    Create a SQLAlchemy Engine with sensible defaults for Streamlit Cloud.
    DATABASE_URL should be set in Streamlit Secrets.
    """
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
    q = text(f'SELECT DISTINCT {col} FROM public.v_finance_logic WHERE {col} IS NOT NULL ORDER BY {col}')
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=3600)
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()
KNOWN_FUNC_CODES = get_distinct("func_code")  # for explicit parsing if user writes "function X"

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

    # higher priority intents first
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
    if any(w in ql for w in ["net", "balance"]):
        return "net"
    if any(w in ql for w in ["deposit", "received", "credited"]):
        return "deposit"
    return "search"

def parse_month_range(q: str, default_year: int | None = None):
    ql = q.lower()
    default_year = default_year or date.today().year

    month_pattern = r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
    found = re.findall(month_pattern, ql)
    found = [m.lower() for m in found if m.lower() in MONTHS]

    if len(found) >= 2:
        m1, m2 = MONTHS[found[0]], MONTHS[found[1]]
        y1 = default_year
        y2 = default_year

        if m2 < m1:
            y2 = default_year + 1

        start = date(y1, m1, 1)
        if m2 == 12:
            end_excl = date(y2 + 1, 1, 1)
        else:
            end_excl = date(y2, m2 + 1, 1)
        return start, end_excl

    return None, None

def infer_date_sql(q: str):
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
    """
    If user explicitly writes something like:
      'function revenue' or 'func_code revenue'
    then return the best matching func_code.
    Otherwise return None.
    """
    ql = q.lower()
    m = re.search(r"\b(?:function|func_code|func)\s*[:=]?\s*([a-z0-9_\-\s]+)", ql)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None

    # Try close match against known func_codes
    matches = get_close_matches(raw.title(), KNOWN_FUNC_CODES, n=1, cutoff=0.6)
    return matches[0] if matches else raw.title()

def func_override_for_intent(intent: str):
    if intent == "revenue":
        return "Revenue"     # force revenue in SQL
    if intent in ("expense", "recoup"):
        return None          # ignore func_code in SQL
    return USE_UI            # normal UI behavior

def apply_intent_func_override(intent: str, question: str, ui_func_code: str) -> str | None:
    """
    Returns:
      - 'Revenue' forced for revenue intent (unless user explicitly sets different function)
      - None to IGNORE func_code filter for expense/recoup/cashflow/trial_balance/search
      - Or a specific func_code if user explicitly requested it in the question
    """
    explicit = parse_explicit_func_code(question)

    # If user explicitly asked for a function, respect it.
    if explicit:
        return explicit

    if intent == "revenue":
        return "Revenue"  # force
    if intent in ("expense", "recoup", "cashflow", "trial_balance", "net", "deposit", "search"):
        return None        # ignore func filter for these intents

    # fallback: respect UI
    return None if ui_func_code == "ALL" else ui_func_code

# -------------------------------------------------
# FILTERS (UI)
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

USE_UI = object()   # sentinel (cannot clash with real values)

def build_ui_where(
    *,
    df,
    dt,
    bank,
    head,
    account,
    func_code,
    func_override=USE_UI
):
    where = ['"date" BETWEEN :df AND :dt']
    params = {"df": df, "dt": dt}

    if bank != "ALL":
        where.append("bank = :bank")
        params["bank"] = bank

    if head != "ALL":
        where.append("head_name = :head")
        params["head"] = head

    if account != "ALL":
        where.append("account = :account")
        params["account"] = account

    # ---------- FUNCTION CODE LOGIC (SQL ONLY) ----------
    if func_override is USE_UI:
        effective_func = func_code        # normal UI behavior
    elif func_override in (None, "ALL"):
        effective_func = "ALL"             # ignore func_code
    else:
        effective_func = func_override     # forced by intent

    if effective_func != "ALL":
        where.append("func_code = :func_code")
        params["func_code"] = effective_func
    # ---------------------------------------------------

    return where, params, effective_func

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_rev, tab_exp, tab_cf, tab_tb, tab_qa = st.tabs(
    ["Revenue", "Expense", "Cashflow", "Trial Balance", "AI Q&A"]
)

# Revenue
with tab_rev:
    st.subheader("Revenue (Monthly)")
    where, params, _eff_func = build_ui_where(
        df=df, dt=dt,
        bank=bank,
        head=head,
        account=account,
        func_code=func_code,
        func_override="Revenue"
    )

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

# Expense
with tab_exp:
    st.subheader("Expenses (Monthly)")
    # DO NOT force Revenue here; use UI filters normally
    where, params, _eff_func = build_ui_where(
        df=df, dt=dt,
        bank=bank,
        head=head,
        account=account,
        func_code=func_code
    )

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

# Cashflow
with tab_cf:
    st.subheader("Cashflow Summary (By Bank & Direction)")
    where, params, _eff_func = build_ui_where(
        df=df, dt=dt,
        bank=bank,
        head=head,
        account=account,
        func_code=func_code,
        func_override=None
    )

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
        st.success(f"Inflow: {inflow:,.0f} PKR  |  Outflow: {abs(outflow):,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR")
    else:
        st.info("No rows found for selected filters/date range.")

# Trial Balance
with tab_tb:
    st.subheader("Trial Balance (As of To Date)")

    # For TB, do not force Revenue; allow UI func_code if set
    tb_where = ["\"date\" <= :dt"]
    tb_params = {"dt": dt}

    if bank != "ALL":
        tb_where.append("bank = :bank"); tb_params["bank"] = bank
    if head != "ALL":
        tb_where.append("head_name = :head"); tb_params["head"] = head
    if account != "ALL":
        tb_where.append("account = :account"); tb_params["account"] = account
    if func_code != "ALL":
        tb_where.append("func_code = :func_code"); tb_params["func_code"] = func_code

    sql = f"""
    select
      account,
      sum(signed_amount) as balance
    from public.v_finance_logic
    where {' and '.join(tb_where)}
    group by 1
    order by 1
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), tb_params).fetchall()

    if rows:
        df_tb = pd.DataFrame(rows, columns=["Account", "Balance"])
        st.dataframe(df_tb, use_container_width=True)
        st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
    else:
        st.info("No rows found for trial balance with current filters.")

# -------------------------------------------------
# AI Q&A
# -------------------------------------------------
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic + Search)")

    st.caption(
        "Examples: How much expense from July to January | Revenue per month | Trial balance as of date | Recoup pending amount | Fuel expense in Dec"
    )

    q = st.text_input("Ask anything…", placeholder="Revenue by head | Pending recoup amount | Which account has highest expense?")

    if q:
        ql = q.lower()
        intent = detect_intent(q)
        payee = extract_payee(q)
        func_override = func_override_for_intent(intent)

        # 🔥 fix: override func_code based on intent, unless explicitly asked in question
        override_func = apply_intent_func_override(intent, q, func_code)

        # Build WHERE using override_func
        where, params, effective_func = build_ui_where(
            df=df, dt=dt,
            bank=bank,
            head=head,
            account=account,
            func_code=func_code,
            func_override=func_override
        )
        where_sql = " AND ".join(where)

        # date inference overrides UI date filter
        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            where = [w for w in where if "between :df and :dt" not in w and "\"date\" between" not in w]
            where.insert(0, date_sql)
            params.update(date_params)

        # month range parsing overrides date filter
        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [w for w in where if "between :df and :dt" not in w and "\"date\" between" not in w and "\"date\" >=" not in w and "\"date\" <" not in w]
            where.insert(0, "\"date\" >= :m_start and \"date\" < :m_end")
            params["m_start"] = m_start
            params["m_end"] = m_end_excl

        if payee:
            where.append("pay_to ilike :payee")
            params["payee"] = f"%{payee}%"

        where_sql = " and ".join(where)

        # ---------- intent-specific routing ----------
        label = ""
        sql = ""

        # Revenue sub-modes
        if intent == "revenue":
            by_head = ("by head" in ql) or ("head wise" in ql) or ("head-wise" in ql)
            by_bank = ("by bank" in ql) or ("bank wise" in ql) or ("bank-wise" in ql)
            per_month = ("per month" in ql) or ("monthly" in ql) or ("trend" in ql) or ("month" in ql)

            if by_head:
                sql = f"""
                select head_name, sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 2 desc
                limit 50
                """
                label = "Revenue by Head"
            elif by_bank:
                sql = f"""
                select coalesce(bank,'UNKNOWN') as bank, sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 2 desc
                """
                label = "Revenue by Bank"
            elif per_month:
                sql = f"""
                select date_trunc('month',"date") as month, sum(coalesce(revenue_amount,0)) as revenue
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                group by 1
                order by 1
                """
                label = "Monthly Revenue"
            else:
                sql = f"""
                select coalesce(sum(coalesce(revenue_amount,0)),0)
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='revenue'
                """
                label = "Total Revenue"

        # Expense sub-modes
        elif intent == "expense":
            if "by head" in q.lower() or "head wise" in q.lower() or "head-wise" in q.lower():
                sql = f"""
                select head_name, coalesce(sum(coalesce(expense_amount,0)),0) as expense
                from public.v_finance_logic
                where {where_sql}
                and entry_type='expense'
                group by 1
                order by 2 desc
                limit 50
                """
                label = "Expense by Head"
            else:
                sql = f"""
                select coalesce(sum(coalesce(expense_amount,0)),0)
                from public.v_finance_logic
                where {where_sql}
                and entry_type='expense'
                """
                label = "Total Expense"

        # Recoup sub-modes
        elif intent == "recoup":
            pending = ("pending" in ql) or ("outstanding" in ql) or ("not recouped" in ql)
            recouped = ("recouped" in ql) or ("settled" in ql)

            if pending:
                sql = f"""
                select coalesce(sum(coalesce(recoup_pending_amount,0)),0)
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='pending'
                """
                label = "Pending Recoup Amount"
            elif recouped:
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0)
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                  and recoup_state='recouped'
                """
                label = "Recouped Total"
            else:
                sql = f"""
                select coalesce(sum(abs(signed_amount)),0)
                from public.v_finance_logic
                where {where_sql}
                  and entry_type='recoup'
                """
                label = "Recoup Total"

        elif intent == "cashflow":
            sql = f"""
            select
              coalesce(bank,'UNKNOWN') as bank,
              direction,
              sum(signed_amount) as amount
            from public.v_finance_logic
            where {where_sql}
            group by 1,2
            order by 1,2
            """
            label = "Cashflow"

        elif intent == "trial_balance":
            sql = f"""
            select account, sum(signed_amount) as balance
            from public.v_finance_logic
            where {where_sql}
            group by 1
            order by 1
            """
            label = "Trial Balance"

        else:
            params["q"] = q
            sql = f"""
            select coalesce(sum(signed_amount),0)
            from public.v_finance_logic
            where {where_sql}
              and (
                search_text % :q
                or search_tsv @@ plainto_tsquery('simple', :q)
              )
            """
            label = "Total (Search matched)"

        # ---------- run + render ----------
        with engine.connect() as conn:
            if intent in ("cashflow", "trial_balance") or ("group by" in sql.lower()):
                rows = conn.execute(text(sql), params).fetchall()

                if not rows:
                    st.warning("No rows found for this question with current filters.")
                else:
                    # Show grouped outputs as tables
                    if label in ("Cashflow", "Trial Balance"):
                        if label == "Cashflow":
                            df_out = pd.DataFrame(rows, columns=["Bank", "Direction", "Amount"])
                            st.dataframe(df_out, use_container_width=True)
                            inflow = df_out[df_out["Direction"] == "in"]["Amount"].sum()
                            outflow = df_out[df_out["Direction"] == "out"]["Amount"].sum()
                            st.success(f"Inflow: {inflow:,.0f} PKR | Outflow: {abs(outflow):,.0f} PKR | Net: {(inflow+outflow):,.0f} PKR")
                        else:
                            df_out = pd.DataFrame(rows, columns=["Account", "Balance"])
                            st.dataframe(df_out, use_container_width=True)
                            st.success(f"Net (sum of balances): {df_out['Balance'].sum():,.0f} PKR")
                    else:
                        # Generic grouped output
                        df_out = pd.DataFrame(rows)
                        # Try to name columns based on label patterns
                        if label in ("Revenue by Head", "Expense by Account"):
                            df_out.columns = ["Key", "Amount"]
                        elif label in ("Revenue by Bank",):
                            df_out.columns = ["Key", "Amount"]
                        elif label in ("Monthly Revenue", "Monthly Expense"):
                            df_out.columns = ["Month", "Amount"]
                        elif label in ("Top Expense Accounts",):
                            df_out.columns = ["Account", "Expense"]

                        st.dataframe(df_out, use_container_width=True)

                        # friendly totals
                        if "Amount" in df_out.columns:
                            st.success(f"{label} (Total): {df_out['Amount'].sum():,.0f} PKR")
                        elif "Expense" in df_out.columns:
                            st.success(f"{label} (Total): {df_out['Expense'].sum():,.0f} PKR")
            else:
                val = conn.execute(text(sql), params).scalar() or 0
                st.success(f"{label}: {val:,.0f} PKR")

        # ---------- explain ----------
        with st.expander("🔍 Why this result?"):
            st.write(f"Intent detected: `{intent}`")
            if payee:
                st.write(f"Payee: `{payee}`")
            if m_start and m_end_excl:
                st.write(f"Month range: `{m_start}` to `{m_end_excl}` (end exclusive)")
            st.write("Filters applied:")
            st.write(f"- Bank: `{bank}`  |  Head: `{head}`  |  Account: `{account}`  |  Function: `{func_code}`")
            st.write(f"- From: `{df}`  |  To: `{dt}`")

# =================================================
# PATCH: Safe override support without replacing UI filters
# =================================================

# Preserve UI filters and only add safe override logic

def build_ui_where_safe(override_func=None):
    """
    Same as build_ui_where but does NOT replace UI filters.
    Only overrides func_code if override_func is provided.
    """
    where = []
    params = {'df': df, 'dt': dt}
    where.append("\"date\" between :df and :dt")

    if bank != 'ALL':
        where.append("bank = :bank"); params['bank'] = bank
    if head != 'ALL':
        where.append("head_name = :head"); params['head'] = head
    if account != 'ALL':
        where.append("account = :account"); params['account'] = account

    effective_func = func_code
    if override_func not in (None, 'ALL'):
        effective_func = override_func

    if effective_func != 'ALL':
        where.append("func_code = :func_code")
        params['func_code'] = effective_func

    return where, params, effective_func

# ---------------- AI override hook ----------------
def get_effective_func(intent, question):
    ql = question.lower()
    if intent == 'revenue':
        return 'Revenue'
    if intent in ('expense','recoup','cashflow','trial_balance','search','net','deposit'):
        return None
    return None