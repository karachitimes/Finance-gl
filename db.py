import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

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

def resolve_relation(sql: str, rel: str) -> str:
    """Replace legacy table reference with chosen relation."""
    return sql.replace("public.gl_register", rel)

def run_scalar(engine, sql: str, params: dict, *, rel: str) -> float:
    sql = resolve_relation(sql, rel)
    with engine.connect() as conn:
        v = conn.execute(text(sql), params).scalar()
    return float(v or 0)

def run_df(engine, sql: str, params: dict, columns: list[str] | None = None, *, rel: str) -> pd.DataFrame:
    sql = resolve_relation(sql, rel)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    df_out = pd.DataFrame(rows)
    if columns and not df_out.empty:
        df_out.columns = columns
    return df_out
