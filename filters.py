import streamlit as st
from datetime import date
from sqlalchemy import text

from db import get_engine

# -------------------------------------------------
# Source relation detection (schema-safe)
# NOTE: Do NOT pass SQLAlchemy engine into st.cache_data (unhashable).
# -------------------------------------------------

@st.cache_data(ttl=300)
def get_source_relation_cached() -> str:
    engine = get_engine()
    with engine.connect() as conn:
        try:
            conn.execute(text("select 1 from public.v_finance_semantic limit 1"))
            return "public.v_finance_semantic"
        except Exception:
            return "public.gl_register"

def get_source_relation(engine) -> str:
    # Keep signature so callers don't change; engine intentionally unused here.
    return get_source_relation_cached()


# -------------------------------------------------
# Distinct lists for filters (cached)
# -------------------------------------------------

@st.cache_data(ttl=3600)
def get_distinct_cached(rel: str, col: str) -> list[str]:
    engine = get_engine()
    q = text(f'SELECT DISTINCT {col} FROM {rel} WHERE {col} IS NOT NULL ORDER BY 1')
    with engine.connect() as conn:
        return [str(r[0]) for r in conn.execute(q).fetchall()]

def get_distinct(engine, rel: str, col: str) -> list[str]:
    # Keep signature; engine unused for caching safety.
    return get_distinct_cached(rel, col)

@st.cache_data(ttl=3600)
def get_distinct_years_cached(rel: str) -> list[int]:
    engine = get_engine()
    q = text(f'SELECT DISTINCT EXTRACT(YEAR FROM "date")::int AS year FROM {rel} ORDER BY year')
    with engine.connect() as conn:
        return [int(r[0]) for r in conn.execute(q).fetchall()]

def get_distinct_years(engine, rel: str) -> list[int]:
    return get_distinct_years_cached(rel)


# -------------------------------------------------
# WHERE builder
# -------------------------------------------------
USE_UI = object()  # sentinel

def build_where_from_ui(
    df, dt, bank, head, account, attribute, func_code,
    *, fy_label: str | None = None, func_override=USE_UI
):
    where = []
    params: dict[str, object] = {}

    # Defensive: normalize None to 'ALL' (keeps behavior unchanged)
    func_code = func_code or "ALL"

    if fy_label and fy_label != "ALL":
        try:
            start_year = int(fy_label.replace("FY","").split("-")[0])
            fy_start = date(start_year, 7, 1)
            fy_end   = date(start_year + 1, 6, 30)
            where.append('"date" >= :fy_start'); params["fy_start"] = fy_start
            where.append('"date" <= :fy_end');   params["fy_end"] = fy_end
        except Exception:
            where.append('"date" between :df and :dt'); params["df"]=df; params["dt"]=dt
    else:
        where.append('"date" between :df and :dt'); params["df"]=df; params["dt"]=dt

    if bank != "ALL":
        where.append("bank = :bank"); params["bank"] = bank
    if head != "ALL":
        where.append("head_name = :head_name"); params["head_name"] = head
    if account != "ALL":
        where.append("account = :account"); params["account"] = account
    if attribute != "ALL":
        where.append("attribute = :attribute"); params["attribute"] = attribute

    # func_code resolve
    if func_override is USE_UI:
        effective_func = func_code if func_code != "ALL" else None
    elif func_override in (None,"ALL"):
        effective_func = None
    else:
        effective_func = func_override

    if effective_func is not None:
        where.append("func_code = :func_code"); params["func_code"] = effective_func

    return where, params, (effective_func if effective_func is not None else "ALL")


# -------------------------------------------------
# UI render
# -------------------------------------------------
def ensure_default_state():
    # Set defaults individually (avoids partial state bugs and widget-key conflicts)
    if "filters_applied" not in st.session_state:
        st.session_state["filters_applied"] = False
    if "df" not in st.session_state:
        st.session_state["df"] = date(2025,1,1)
    if "dt" not in st.session_state:
        st.session_state["dt"] = date.today()
    if "bank" not in st.session_state:
        st.session_state["bank"] = "ALL"
    if "head" not in st.session_state:
        st.session_state["head"] = "ALL"
    if "account" not in st.session_state:
        st.session_state["account"] = "ALL"
    if "attribute" not in st.session_state:
        st.session_state["attribute"] = "ALL"
    if "func_code" not in st.session_state:
        st.session_state["func_code"] = "ALL"
    if "fy_label" not in st.session_state:
        st.session_state["fy_label"] = "ALL"

def render_filter_bar(engine):
    ensure_default_state()
    rel = get_source_relation(engine)

    with st.form(key="filter_form"):
        c1, c2 = st.columns(2)
        with c1:
            new_df = st.date_input("From Date", value=st.session_state["df"])
        with c2:
            new_dt = st.date_input("To Date", value=st.session_state["dt"])

        banks = ["ALL"] + get_distinct(engine, rel, "bank")
        heads = ["ALL"] + get_distinct(engine, rel, "head_name")
        accounts = ["ALL"] + get_distinct(engine, rel, "account")
        try:
            attrs = get_distinct(engine, rel, "attribute")
        except Exception:
            attrs = []
        attrs_list = ["ALL"] + sorted(attrs)
        funcs = ["ALL"] + get_distinct(engine, rel, "func_code")

        years = get_distinct_years(engine, rel)
        fy_options = ["ALL"] + [f"FY{y}-{(y+1)%100:02d}" for y in years]

        def _idx(lst, v):
            return lst.index(v) if v in lst else 0

        new_bank = st.selectbox("Bank", banks, index=_idx(banks, st.session_state["bank"]))
        new_head = st.selectbox("Head", heads, index=_idx(heads, st.session_state["head"]))
        new_account = st.selectbox("Account", accounts, index=_idx(accounts, st.session_state["account"]))
        new_attribute = st.selectbox("Attribute", attrs_list, index=_idx(attrs_list, st.session_state["attribute"]))
        new_func_code = st.selectbox("Function Code", funcs, index=_idx(funcs, st.session_state["func_code"]))
        new_fy_label = st.selectbox("Fiscal Year", fy_options, index=_idx(fy_options, st.session_state["fy_label"]))

        apply_filters = st.form_submit_button("Apply Filters")

    if apply_filters or not st.session_state["filters_applied"]:
        st.session_state["filters_applied"] = True
        st.session_state["df"] = new_df
        st.session_state["dt"] = new_dt
        st.session_state["bank"] = new_bank
        st.session_state["head"] = new_head
        st.session_state["account"] = new_account
        st.session_state["attribute"] = new_attribute
        st.session_state["func_code"] = new_func_code
        st.session_state["fy_label"] = new_fy_label

    return {
        "df": st.session_state["df"],
        "dt": st.session_state["dt"],
        "bank": st.session_state["bank"],
        "head": st.session_state["head"],
        "account": st.session_state["account"],
        "attribute": st.session_state["attribute"],
        "func_code": st.session_state["func_code"],
        "fy_label": st.session_state["fy_label"],
    }
