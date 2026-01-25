import streamlit as st
from sqlalchemy import create_engine, text
from datetime import date
from sqlalchemy.exc import OperationalError

from difflib import get_close_matches
import re


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Finance System", layout="wide")

DATABASE_URL = st.secrets["DATABASE_URL"]




# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------
def get_engine():
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
# HELPERS
# -------------------------------------------------
# Render UI first, then connect
st.title("📊 Finance Analytics System")

try:
    ok = test_connection(engine)
    st.success("Database connected ✅")

except OperationalError as e:
    st.error("Database connection failed")

    try:
        real_err = str(e.orig)
    except:
        real_err = str(e)

    st.code(real_err)   # 👈 THIS shows the real PostgreSQL error
    st.stop()

@st.cache_data
def get_distinct(col):
    q = text(f"SELECT DISTINCT {col} FROM v_finance_logic WHERE {col} IS NOT NULL ORDER BY {col}")
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(q).fetchall()]

@st.cache_data
def get_known_payees():
    return get_distinct("pay_to")

KNOWN_PAYEES = get_known_payees()

def best_payee_match(name):
    if not name:
        return None
    matches = get_close_matches(name.title(), KNOWN_PAYEES, n=1, cutoff=0.75)
    return matches[0] if matches else name.title()

# -------------------------------------------------
# SEMANTIC SUBJECT EXTRACTION (FIXED)
# -------------------------------------------------
def extract_semantic_subject(q: str):
    q = q.lower()

    noise = [
        "how much", "total", "paid", "payment",
        "with", "month", "months", "monthly", "for",
        "last", "this", "current", "on", "which", "date", "wise"
    ]

    for n in noise:
        q = q.replace(n, " ")

    q = re.sub(r"\s+", " ", q).strip()
    return q if len(q) >= 3 else None

# -------------------------------------------------
# DATE RANGE INFERENCE
# -------------------------------------------------
def infer_date_range(q: str):
    q = q.lower()

    if "last month" in q:
        return "date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' AND date < DATE_TRUNC('month', CURRENT_DATE)"

    if "this month" in q:
        return "date >= DATE_TRUNC('month', CURRENT_DATE) AND date < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'"

    return None

# -------------------------------------------------
# QUESTION PARSER
# -------------------------------------------------
def parse_question(q):
    ql = q.lower()

    res = {
        "intent": "total",        # total | monthly | net | deposit | expense
        "pay_to": None,
        "semantic_subject": extract_semantic_subject(q)
    }

    if any(w in ql for w in ["month wise", "with months", "monthly"]):
        res["intent"] = "monthly"

    if any(w in ql for w in ["net", "balance"]):
        res["intent"] = "net"

    if any(w in ql for w in ["deposit", "received", "credited"]):
        res["intent"] = "deposit"

    if any(w in ql for w in ["paid for", "expense", "charges", "fuel", "repair"]):
        res["intent"] = "expense"

    m = re.search(r"(?:to|by)\s+([a-z\s]+?)(?:\s+with|\s+month|\s+for|$)", ql)
    if m:
        res["pay_to"] = best_payee_match(m.group(1))

    return res

# -------------------------------------------------
# UI FILTERS
# -------------------------------------------------
st.title("📊 Finance Analytics System")

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
# QUESTION BOX
# -------------------------------------------------
st.divider()
st.subheader("💬 Ask a Finance Question")

q = st.text_input(
    "Ask anything…",
    placeholder="paid for generator | total deposit by mukhi | net balance for power"
)

if q:
    parsed = parse_question(q)

    where = []
    params = {}
    matched_columns = []

    # DATE FILTER
    date_sql = infer_date_range(q)
    if date_sql:
        where.append(date_sql)
    else:
        where.append("date BETWEEN :df AND :dt")
        params["df"] = df
        params["dt"] = dt

    # UI FILTERS
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

    # AMOUNT COLUMN
    if parsed["intent"] == "deposit":
        amount_field = "credit_deposit"
        where.append("credit_deposit > 0")
    else:
        amount_field = "debit_payment"
        where.append("debit_payment > 0")

    # SEMANTIC MATCH
    if parsed["semantic_subject"]:
        if parsed["intent"] == "expense":
            where.append("description ILIKE :term")
            matched_columns.append("description")
        else:
            where.append("""
                (
                    pay_to ILIKE :term
                 OR description ILIKE :term
                 OR head_name ILIKE :term
                 OR account ILIKE :term
                )
            """)
            matched_columns.extend(["pay_to", "description", "head_name", "account"])

        params["term"] = f"%{parsed['semantic_subject']}%"

    if parsed["pay_to"]:
        where.append("pay_to ILIKE :p")
        params["p"] = f"%{parsed['pay_to']}%"
        matched_columns.append("pay_to")

    where_sql = " AND ".join(where)

    with engine.connect() as conn:

        # -----------------------------
        # NET BALANCE
        # -----------------------------
        if parsed["intent"] == "net":
            sql = f"""
                SELECT
                    COALESCE(SUM(debit_payment),0),
                    COALESCE(SUM(credit_deposit),0)
                FROM v_finance_logic
                WHERE {where_sql}
            """
            paid, received = conn.execute(text(sql), params).fetchone()
            st.success(f"Net Balance : {paid - received:,.0f} PKR")
            st.caption(f"Paid: {paid:,.0f} | Received: {received:,.0f}")

        # -----------------------------
        # MONTHLY TREND
        # -----------------------------
        elif parsed["intent"] == "monthly":
            sql = f"""
                SELECT DATE_TRUNC('month', date), SUM({amount_field})
                FROM v_finance_logic
                WHERE {where_sql}
                GROUP BY 1
                ORDER BY 1
            """
            rows = conn.execute(text(sql), params).fetchall()

            if rows:
                st.subheader("Monthly Breakdown")
                st.dataframe(
                    [{"Month": r[0].strftime("%Y-%m"), "Amount": float(r[1])} for r in rows],
                    use_container_width=True
                )
                total_val = sum(float(r[1]) for r in rows)
                st.success(f"Total : {total_val:,.0f} PKR")

                st.subheader("📈 Trend")
                st.line_chart({r[0].strftime("%Y-%m"): float(r[1]) for r in rows})

        # -----------------------------
        # TOTAL / DEPOSIT / EXPENSE
        # -----------------------------
        else:
            sql = f"""
                SELECT COALESCE(SUM({amount_field}),0)
                FROM v_finance_logic
                WHERE {where_sql}
            """
            val = conn.execute(text(sql), params).scalar() or 0
            st.success(f"Total : {val:,.0f} PKR")


        # -----------------------------
        # EXPLANATION
        # -----------------------------
        with st.expander("🔍 Why this result?"):
            st.write(f"Intent: `{parsed['intent']}`")
            if parsed["semantic_subject"]:
                st.write(f"Subject: `{parsed['semantic_subject']}`")
            if parsed["pay_to"]:
                st.write(f"Payee: `{parsed['pay_to']}`")
            st.write("Data Columns:")
            st.write("• debit_payment (expenses)")
            st.write("• credit_deposit (receipts)")

st.divider()
