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
    DATABASE_URL should be set in Streamlit Secrets, e.g.
    postgresql+psycopg2://user:pass@...:6543/postgres?sslmode=require
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
    # col is controlled by code (not user), safe to interpolate.
    q = text(f"SELECT DISTINCT {col} FROM public.v_finance_logic WHERE {col} IS NOT NULL ORDER BY {col}")
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data(ttl=3600)
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()

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
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}  # jan..dec => 1..12

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
    if any(w in ql for w in ["monthly", "month wise", "month-wise", "per month"]):
        return "monthly"
    return "search"

def parse_month_range(q: str, default_year: int | None = None):
    """
    Parses phrases like "July to January", "jul - jan".
    Uses default_year (current year) when year is not specified.
    Returns (start_date, end_date_exclusive) or (None, None).
    """
    ql = q.lower()
    default_year = default_year or date.today().year

    month_pattern = r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
    found = re.findall(month_pattern, ql)
    found = [m.lower() for m in found if m.lower() in MONTHS]

    if len(found) >= 2:
        m1, m2 = MONTHS[found[0]], MONTHS[found[1]]
        y1 = default_year
        y2 = default_year

        # if range wraps year boundary (e.g., jul -> jan), advance end year
        if m2 < m1:
            y2 = default_year + 1

        start = date(y1, m1, 1)
        # end exclusive = first day of month after m2
        if m2 == 12:
            end_excl = date(y2 + 1, 1, 1)
        else:
            end_excl = date(y2, m2 + 1, 1)
        return start, end_excl

    return None, None

def infer_date_sql(q: str):
    """
    Returns (sql_fragment, params_dict) or (None, {}).
    Uses server CURRENT_DATE for relative phrases.
    """
    ql = q.lower()

    if "last month" in ql:
        return (
            "date >= date_trunc('month', current_date) - interval '1 month' "
            "and date < date_trunc('month', current_date)",
            {}
        )
    if "this month" in ql:
        return (
            "date >= date_trunc('month', current_date) "
            "and date < date_trunc('month', current_date) + interval '1 month'",
            {}
        )
    return None, {}

def extract_payee(q: str):
    ql = q.lower()
    m = re.search(r"(?:to|by)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        return best_payee_match(m.group(1))
    return None

# -------------------------------------------------
# FILTERS
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

def build_ui_where():
    where = []
    params = {"df": df, "dt": dt}
    where.append("date between :df and :dt")

    if bank != "ALL":
        where.append("bank = :bank")
        params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head")
        params["head"] = head
    if account != "ALL":
        where.append("account = :account")
        params["account"] = account
    if func_code != "ALL":
        where.append("func_code = :func_code")
        params["func_code"] = func_code

    return where, params

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_rev, tab_exp, tab_cf, tab_tb, tab_qa = st.tabs(
    ["Revenue", "Expense", "Cashflow", "Trial Balance", "AI Q&A"]
)

# Revenue
with tab_rev:
    st.subheader("Revenue (Monthly)")
    where, params = build_ui_where()

    sql = f"""
    select date_trunc('month', date) as month,
           sum(signed_amount) as revenue
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
    where, params = build_ui_where()

    sql = f"""
    select date_trunc('month', date) as month,
           sum(abs(signed_amount)) as expense
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
    where, params = build_ui_where()

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
        outflow = df_cf[df_cf["Direction"] == "out"]["Amount"].sum()
        st.success(f"Inflow: {inflow:,.0f} PKR  |  Outflow: {outflow:,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR")
    else:
        st.info("No rows found for selected filters/date range.")

# Trial Balance
with tab_tb:
    st.subheader("Trial Balance (As of To Date)")
    params = {"dt": dt}
    where = ["date <= :dt"]

    if bank != "ALL":
        where.append("bank = :bank")
        params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head")
        params["head"] = head
    if account != "ALL":
        where.append("account = :account")
        params["account"] = account
    if func_code != "ALL":
        where.append("func_code = :func_code")
        params["func_code"] = func_code

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

# AI Q&A
with tab_qa:
    st.subheader("Ask a Finance Question (Deterministic + Search)")

    st.caption(
        "Examples: How much expense from July to January | Revenue per month | Trial balance as of date | Recoup pending amount | Fuel expense in Dec"
    )

    q = st.text_input("Ask anything…", placeholder="How much expense from July to January?")

    if q:
        intent = detect_intent(q)
        payee = extract_payee(q)

        where, params = build_ui_where()

        date_sql, date_params = infer_date_sql(q)
        if date_sql:
            where = [w for w in where if "date between" not in w]
            where.insert(0, date_sql)
            params.update(date_params)

        m_start, m_end_excl = parse_month_range(q)
        if m_start and m_end_excl:
            where = [w for w in where if "date between" not in w and "date >=" not in w and "date <" not in w]
            where.insert(0, "date >= :m_start and date < :m_end")
            params["m_start"] = m_start
            params["m_end"] = m_end_excl

        if payee:
            where.append("pay_to ilike :payee")
            params["payee"] = f"%{payee}%"

        where_sql = " and ".join(where)

        if intent == "revenue":
            sql = f"""
            select coalesce(sum(signed_amount),0)
            from public.v_finance_logic
            where {where_sql}
              and entry_type='revenue'
            """
            label = "Total Revenue"

        elif intent == "expense":
            sql = f"""
            select coalesce(sum(abs(signed_amount)),0)
            from public.v_finance_logic
            where {where_sql}
              and entry_type='expense'
            """
            label = "Total Expense"

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
            tb_where = ["date <= :dt"]
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
            select account, sum(signed_amount) as balance
            from public.v_finance_logic
            where {' and '.join(tb_where)}
            group by 1
            order by 1
            """
            label = "Trial Balance"
            params = tb_params

        elif intent == "recoup":
            sql = f"""
            select coalesce(sum(abs(signed_amount)),0)
            from public.v_finance_logic
            where {where_sql}
              and entry_type='recoup'
            """
            label = "Recoup Total"

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

        with engine.connect() as conn:
            if intent in ("cashflow", "trial_balance"):
                rows = conn.execute(text(sql), params).fetchall()
                if intent == "cashflow":
                    if rows:
                        df_out = pd.DataFrame(rows, columns=["Bank", "Direction", "Amount"])
                        st.dataframe(df_out, use_container_width=True)
                        inflow = df_out[df_out["Direction"] == "in"]["Amount"].sum()
                        outflow = df_out[df_out["Direction"] == "out"]["Amount"].sum()
                        st.success(f"Inflow: {inflow:,.0f} PKR | Outflow: {outflow:,.0f} PKR | Net: {(inflow+outflow):,.0f} PKR")
                    else:
                        st.warning("No rows found for this cashflow question.")
                else:
                    if rows:
                        df_out = pd.DataFrame(rows, columns=["Account", "Balance"])
                        st.dataframe(df_out, use_container_width=True)
                        st.success(f"Net (sum of balances): {df_out['Balance'].sum():,.0f} PKR")
                    else:
                        st.warning("No rows found for this trial balance question.")
            else:
                val = conn.execute(text(sql), params).scalar() or 0
                st.success(f"{label}: {val:,.0f} PKR")

            confidence = None
            if intent == "search":
                conf_sql = f"""
                select greatest(
                    coalesce(max(similarity(search_text, :q)), 0),
                    coalesce(max(ts_rank(search_tsv, plainto_tsquery('simple', :q))), 0)
                ) as score
                from public.v_finance_logic
                where {where_sql}
                  and (
                    search_text % :q
                    or search_tsv @@ plainto_tsquery('simple', :q)
                  )
                """
                score = conn.execute(text(conf_sql), params).scalar() or 0
                confidence = round(min(float(score), 1.0) * 100)

        with st.expander("🔍 Why this result?"):
            st.write(f"Intent detected: `{intent}`")
            if payee:
                st.write(f"Payee: `{payee}`")
            if m_start and m_end_excl:
                st.write(f"Month range: `{m_start}` to `{m_end_excl}` (end exclusive)")
            st.write("Filters applied:")
            st.write(f"- Bank: `{bank}`  |  Head: `{head}`  |  Account: `{account}`  |  Function: `{func_code}`")
            st.write(f"- From: `{df}`  |  To: `{dt}`")
            if confidence is not None:
                st.write(f"Search confidence: **{confidence}%**")
