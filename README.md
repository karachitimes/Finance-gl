# Finance Analytics System (Modular Streamlit App)

This repository is a modularized version of your original `streamlit_app.py`.

## Files
- `app.py` - Streamlit entrypoint (UI + routing)
- `db.py` - database engine and query helpers
- `filters.py` - UI filter bar + where builder
- `semantic.py` - view names + schema-safe SQL expressions
- `dashboards.py` - Revenue/Expense/Cashflow/Trial Balance tabs
- `recoup.py` - Recoup KPI tab (PowerPivot/DAX equivalents)
- `qa.py` - deterministic AI Q&A router
- `utils.py` - dataframe rendering helpers (totals)

## Run locally
1. Create a virtual environment
2. Install requirements
3. Add `.streamlit/secrets.toml` with `DATABASE_URL`
4. Run `streamlit run app.py`
