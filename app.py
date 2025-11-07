import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Proportional Disaggregation", layout="wide")
st.title("Proportional Disaggregation")

st.markdown(
    """
    This application performs **proportional disaggregation** of Controlling data 
    based on capacity ratios from a selected dimension (e.g., Location, Department, GB, etc.).  
    By default, it breaks values by *Location*, but you can flexibly choose other dimensions.
    """,
    unsafe_allow_html=True
)

# =========================
# HELPERS
# =========================
def _clean_cols(df):
    df = df.copy()
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    return df

@st.cache_data
def load_excel(file):
    ext = Path(file.name).suffix.lower()
    engine = "openpyxl" if ext in [".xlsx", ".xlsm"] else None
    xls = pd.ExcelFile(file, engine=engine)
    return {sn: _clean_cols(xls.parse(sn)) for sn in xls.sheet_names}

def normalize_columns(df, norm_maps):
    out = df.copy()
    for col, mapping in norm_maps.items():
        if col in out.columns and isinstance(mapping, dict):
            s = out[col].astype(str)
            out[col] = s.map(mapping).fillna(s)
    return out

def ensure_month_order(df, month_col):
    """Ensure month columns sort chronologically if format is YYYY-MM."""
    try:
        df["_sort_key"] = pd.to_datetime(df[month_col], errors="coerce")
        df = df.sort_values("_sort_key").drop(columns="_sort_key")
    except Exception:
        pass
    return df

def compute_ratio_expanding(df_loc, keys, break_col, cap_col):
    df = df_loc.copy()
    df = df[keys + [break_col, cap_col]]
    cap_by = df.groupby(keys + [break_col], as_index=False)[cap_col].sum()
    cap_total = df.groupby(keys, as_index=False)[cap_col].sum()
    merged = cap_by.merge(cap_total, on=keys, how="left", suffixes=("_dim", "_tot"))
    merged["Ratio"] = merged[f"{cap_col}_dim"] / merged[f"{cap_col}_tot"].replace({0: np.nan})
    merged["Ratio"] = merged["Ratio"].fillna(0.0)
    return merged.drop(columns=[f"{cap_col}_dim", f"{cap_col}_tot"])

def run_fallback_merges_expand(df_ctrl, df_loc, m_loc, m_ctrl, enabled_layers, break_col):
    base = df_ctrl.copy()
    if "Disagg_Value" not in base:
        base["Disagg_Value"] = np.nan

    def _layer(base_df, keys_loc, layer_tag):
        mask = base_df["Disagg_Value"].isna()
        if mask.sum() == 0:
            return base_df, 0
        work = base_df.loc[mask].copy()
        ratio_df = compute_ratio_expanding(df_loc, keys_loc, break_col, m_loc["capacity_loc"])
        left_keys = [m_ctrl[k] for k in keys_loc if k in m_ctrl]
        right_keys = [m_loc[k] for k in keys_loc if k in m_loc]
        work = work.merge(ratio_df, left_on=left_keys, right_on=right_keys, how="left")
        work["Disagg_Value"] = work[m_ctrl["capacity"]] * work["Ratio"]
        cnt = work["Ratio"].notna().sum()
        base_df.loc[mask, "Disagg_Value"] = work["Disagg_Value"]
        return base_df, int(cnt)

    merge_info = {}
    for i, keys_loc in enumerate([
        [m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]],
        [m_loc["dept"], m_loc["emp_type"], m_loc["month"]],
        [m_loc["gb"], m_loc["dept"], m_loc["month"]],
        [m_loc["dept"], m_loc["month"]],
    ], start=1):
        tag = f"L{i}"
        if enabled_layers.get(tag, True):
            base, cnt = _layer(base, keys_loc, tag)
            merge_info[tag] = cnt
    return base, merge_info

def compute_outputs(df, m_ctrl):
    df = df.copy()
    denom_cap = df[m_ctrl["capacity"]].replace({0: np.nan})
    share = df["Disagg_Value"] / denom_cap
    df["Budget_Disagg"] = df[m_ctrl["budget"]] * share.fillna(0)
    denom = (df[m_ctrl["rate"]] * df["Disagg_Value"]).replace({0: np.nan})
    df["Billable_Disagg"] = (df[m_ctrl["budget"]] / denom).fillna(0)
    return df

def guess(colnames, candidates):
    col_lower = {str(c).lower(): c for c in colnames}
    for c in candidates:
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Upload Excel")
    f = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    on_div0 = st.selectbox("Divide-by-zero handling", ["zero", "blank"], index=0)

if not f:
    st.info("Upload Excel to start.")
    st.stop()

# =========================
# LOAD SHEETS
# =========================
sheets = load_excel(f)
sheet_loc = st.selectbox("Location sheet", sheets.keys())
sheet_ctrl = st.selectbox("Controlling sheet", sheets.keys())
loc, ctrl = sheets[sheet_loc], sheets[sheet_ctrl]

# =========================
# BREAK DIMENSION
# =========================
st.markdown("---")
st.header("Break Configuration")
break_col_key = st.selectbox("Select Break Dimension", ["emp_location", "dept", "gb", "emp_type"], index=0)
st.caption(f"Disaggregate by **{break_col_key}** capacity ratios.")

# =========================
# COLUMN MAPPING
# =========================
loc_candidates = {
    "gb": ["GB"],
    "dept": ["Dept", "Department"],
    "emp_type": ["Emp Type", "Service"],
    "month": ["Month"],
    "emp_location": ["Emp Location", "Location"],
    "capacity_loc": ["Capacity_Location", "Cap Location"],
}
ctrl_candidates = {
    "gb": ["GB"],
    "dept": ["Dept", "Department"],
    "emp_type_like": ["Emp Type", "Service"],
    "month": ["Month"],
    "capacity": ["Capacity"],
    "budget": ["Budget"],
    "rate": ["Rate"],
}

loc_map, ctrl_map = {}, {}
c1, c2 = st.columns(2)
with c1:
    st.subheader("Location columns")
    for k, opts in loc_candidates.items():
        col = guess(loc.columns, opts)
        loc_map[k] = st.selectbox(k, [None] + list(loc.columns), index=([None] + list(loc.columns)).index(col) if col else 0)
with c2:
    st.subheader("Controlling columns")
    for k, opts in ctrl_candidates.items():
        col = guess(ctrl.columns, opts)
        ctrl_map[k] = st.selectbox(k, [None] + list(ctrl.columns), index=([None] + list(ctrl.columns)).index(col) if col else 0)

# =========================
# RUN + LAYERS
# =========================
st.markdown("---")
st.header("Run Processing")
cols = st.columns(4)
enabled_layers = {
    "L1": cols[0].checkbox("L1 GB+Dept+EmpType+Month", True),
    "L2": cols[1].checkbox("L2 Dept+EmpType+Month", True),
    "L3": cols[2].checkbox("L3 GB+Dept+Month", True),
    "L4": cols[3].checkbox("L4 Dept+Month", True),
}
run = st.button("Run")
if run:
    merged, info = run_fallback_merges_expand(ctrl, loc, loc_map, ctrl_map, enabled_layers, break_col_key)
    merged = compute_outputs(merged, ctrl_map)
    st.session_state["merged_result"] = merged
    st.session_state["merge_info"] = info
    st.success("Processed successfully!")

# =========================
# RESULTS
# =========================
if "merged_result" in st.session_state:
    df = st.session_state["merged_result"]
    st.subheader("Result Snapshot")
    st.caption(f"Total rows: {len(df)} | Layer counts: {st.session_state['merge_info']}")
    st.dataframe(df.head(50), use_container_width=True)

# =========================
# PIVOT TABLE (Excel-like)
# =========================
st.markdown("---")
st.header("Pivot Table")

if "merged_result" not in st.session_state:
    st.info("Run processing first.")
else:
    df = st.session_state["merged_result"]
    cols_all = df.columns.tolist()

    st.session_state.setdefault("pivot_rows", [ctrl_map.get("gb")])
    st.session_state.setdefault("pivot_cols", [ctrl_map.get("month")])
    st.session_state.setdefault("pivot_vals", ["Disagg_Value"])
    st.session_state.setdefault("pivot_filters", [ctrl_map.get("dept")])
    st.session_state.setdefault("pivot_agg", "sum")

    with st.expander("Pivot configuration", expanded=True):
        agg = st.selectbox("Aggregation", ["sum", "mean", "count"], index=["sum","mean","count"].index(st.session_state["pivot_agg"]))
        rows = st.multiselect("Rows", cols_all, default=st.session_state["pivot_rows"])
        cols = st.multiselect("Columns", cols_all, default=st.session_state["pivot_cols"])
        vals = st.multiselect("Values", cols_all, default=st.session_state["pivot_vals"])
        flt = st.multiselect("Filters", cols_all, default=st.session_state["pivot_filters"])

        # filter value pickers
        flt_vals = {}
        for fc in flt:
            uniq = sorted(map(str, df[fc].dropna().unique().tolist()))
            opts = ["(All)"] + uniq
            sel = st.multiselect(f"{fc} values", opts, default=["(All)"])
            if "(All)" not in sel:
                df = df[df[fc].astype(str).isin(sel)]
            flt_vals[fc] = sel

    # build pivot
    aggfunc = {"sum": np.sum, "mean": np.mean, "count": "count"}[agg]
    pivot_df = pd.pivot_table(df, index=rows or None, columns=cols or None, values=vals or None,
                              aggfunc=aggfunc, fill_value=0, dropna=False)
    pivot_df = pivot_df.reset_index()
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [" | ".join([str(x) for x in tup if x != ""]) for tup in pivot_df.columns.values]

    st.dataframe(pivot_df, use_container_width=True)

    # download
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        pivot_df.to_excel(w, sheet_name="Pivot", index=False)
    st.download_button("Download Pivot Excel", data=buf.getvalue(), file_name="pivot_result.xlsx")

# --- footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;font-size:13px;color:#666;'>Crafted with ❤️ by BGSV/CTG Data Team</div>",
    unsafe_allow_html=True,
)
