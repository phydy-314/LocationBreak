import io
import os
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
    based on capacity ratios from Location-level capacity.  
    Typically, it breaks values by *Employee Location* to allocate Controlling values proportionally.
    """,
    unsafe_allow_html=True
)

# =========================
# CONSTANTS
# =========================
REQUIRED_KEYS_LOCATION = [
    "gb",
    "dept",
    "emp_type",
    "month",
    "emp_location",
    "capacity_loc",
]
REQUIRED_KEYS_CTRL = [
    "gb",
    "dept",
    "emp_type_like",
    "month",
    "capacity",
    "budget",
    "rate",
]

# =========================
# HELPERS
# =========================
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name=None):
    name = getattr(file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"
    elif ext == ".xlsb":
        engine = "pyxlsb"
    else:
        engine = None

    xls = pd.ExcelFile(file, engine=engine)
    if sheet_name is None:
        return {sn: _clean_cols(xls.parse(sn)) for sn in xls.sheet_names}
    else:
        return {sheet_name: _clean_cols(xls.parse(sheet_name))}


def normalize_columns(df: pd.DataFrame, norm_maps: dict) -> pd.DataFrame:
    if not norm_maps:
        return df
    out = df.copy()
    for col, mapping in norm_maps.items():
        if col in out.columns and isinstance(mapping, dict) and mapping:
            s = out[col].astype(str)
            out[col] = s.map(mapping).fillna(s)
    return out


def compute_ratio(df_loc, m_loc):
    """Compute proportional ratio of capacity by location."""
    use_cols = [
        m_loc["gb"],
        m_loc["dept"],
        m_loc["emp_type"],
        m_loc["month"],
        m_loc["emp_location"],
        m_loc["capacity_loc"],
    ]
    df = df_loc[use_cols].copy()

    cap_by_loc = (
        df.groupby(
            [m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"], m_loc["emp_location"]],
            dropna=False,
            as_index=False,
        )[m_loc["capacity_loc"]]
        .sum()
        .rename(columns={m_loc["capacity_loc"]: "_cap_loc"})
    )

    cap_total = (
        df.groupby(
            [m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]],
            dropna=False,
            as_index=False,
        )[m_loc["capacity_loc"]]
        .sum()
        .rename(columns={m_loc["capacity_loc"]: "_cap_total"})
    )

    ratio_df = cap_by_loc.merge(cap_total, on=[m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]], how="left")
    ratio_df["Ratio"] = ratio_df["_cap_loc"] / ratio_df["_cap_total"].replace({0: np.nan})
    ratio_df["Ratio"] = ratio_df["Ratio"].fillna(0.0)
    return ratio_df


def run_disagg(df_ctrl, df_loc, m_loc, m_ctrl):
    ratio_df = compute_ratio(df_loc, m_loc)

    base = df_ctrl.copy()
    base = base.merge(
        ratio_df,
        how="left",
        left_on=[m_ctrl["gb"], m_ctrl["dept"], m_ctrl["emp_type_like"], m_ctrl["month"]],
        right_on=[m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]],
    )
    base["Disagg_Value"] = base[m_ctrl["capacity"]] * base["Ratio"]

    cap_series = base[m_ctrl["capacity"]].replace({0: np.nan})
    share = base["Disagg_Value"] / cap_series
    share = np.nan_to_num(share, nan=0.0)

    base["Budget_Disagg"] = base[m_ctrl["budget"]] * share
    denom = (base[m_ctrl["rate"]] * base["Disagg_Value"]).replace({0: np.nan})
    base["Billable_Disagg"] = base[m_ctrl["budget"]] / denom
    base["Billable_Disagg"] = base["Billable_Disagg"].fillna(0.0)
    return base


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
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xlsm", "xls", "xlsb"])
    if not excel_file:
        st.info("Upload your Excel file to get started.")
        st.stop()

# =========================
# LOAD SHEETS
# =========================
all_sheets = load_excel(excel_file)
sheet_loc = st.selectbox("Location sheet", options=list(all_sheets.keys()), index=0)
sheet_ctrl = st.selectbox("Controlling sheet", options=list(all_sheets.keys()), index=1 if len(all_sheets) > 1 else 0)

df_loc = all_sheets[sheet_loc].copy()
df_ctrl = all_sheets[sheet_ctrl].copy()

# =========================
# COLUMN MAPPING
# =========================
st.markdown("---")
st.header("Map Columns")

loc_map, ctrl_map = {}, {}

loc_candidates = {
    "gb": ["GB"],
    "dept": ["Resource Dept", "Dept", "Department"],
    "emp_type": ["Emp Type", "Header Service", "Service"],
    "month": ["Month", "Revenue Month"],
    "emp_location": ["Emp Location", "Dim Location", "Location"],
    "capacity_loc": ["Capacity_Location", "Capacity Location", "Cap Location"],
}
ctrl_candidates = {
    "gb": ["GB"],
    "dept": ["Resource Dept", "Dept", "Department"],
    "emp_type_like": ["Header Service", "Emp Type", "Service"],
    "month": ["Revenue Month", "Month"],
    "capacity": ["Capacity"],
    "budget": ["Budget"],
    "rate": ["Rate", "Selling Rate"],
}

c1, c2 = st.columns(2)
with c1:
    st.subheader("Location columns")
    for key in REQUIRED_KEYS_LOCATION:
        opts = [None] + list(df_loc.columns)
        g = guess(df_loc.columns, loc_candidates.get(key, []))
        idx = opts.index(g) if g in opts else 0
        loc_map[key] = st.selectbox(f"{key}", options=opts, index=idx, key=f"loc_{key}")
with c2:
    st.subheader("Controlling columns")
    for key in REQUIRED_KEYS_CTRL:
        opts = [None] + list(df_ctrl.columns)
        g = guess(df_ctrl.columns, ctrl_candidates.get(key, []))
        idx = opts.index(g) if g in opts else 0
        ctrl_map[key] = st.selectbox(f"{key}", options=opts, index=idx, key=f"ctrl_{key}")

# =========================
# RUN
# =========================
st.markdown("---")
st.header("Run Disaggregation")

if st.button("Run processing", type="primary"):
    result = run_disagg(df_ctrl, df_loc, loc_map, ctrl_map)
    st.session_state["merged_result"] = result
    st.success("Processing completed successfully!")

if "merged_result" in st.session_state:
    st.subheader("Result Preview")
    st.dataframe(st.session_state["merged_result"].head(50), use_container_width=True)

    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        st.session_state["merged_result"].to_excel(writer, index=False, sheet_name="Result")
    st.download_button(
        "Download Excel Result",
        data=out_buf.getvalue(),
        file_name="disaggregation_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; font-size:13px; color:#666; margin-top:8px;">
        Crafted with care by <strong>BGSV/CTG Data Team</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)
