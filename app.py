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
    based on capacity ratios from employee location.
    Typically, it breaks Controlling values by *Emp Location* proportionally to capacity.
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
def load_excel(file):
    ext = Path(file.name).suffix.lower()
    engine = "openpyxl" if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"] else None
    xls = pd.ExcelFile(file, engine=engine)
    return {sn: _clean_cols(xls.parse(sn)) for sn in xls.sheet_names}


def normalize_columns(df: pd.DataFrame, norm_maps: dict) -> pd.DataFrame:
    if not norm_maps:
        return df
    out = df.copy()
    for col, mapping in norm_maps.items():
        if col in out.columns and isinstance(mapping, dict):
            s = out[col].astype(str)
            out[col] = s.map(mapping).fillna(s)
    return out


def compute_ratio(df_loc, m_loc):
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
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xlsm", "xls"])
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
# NORMALIZATION
# =========================
st.markdown("---")
st.header("Normalization Rules")

st.session_state.setdefault("norm_loc_maps", {})
st.session_state.setdefault("norm_ctrl_maps", {})

tab_loc, tab_ctrl = st.tabs(["Location", "Controlling"])

with tab_loc:
    loc_cols = list(df_loc.columns)
    sel_loc_cols = st.multiselect(
        "Columns to normalize (Location)",
        options=loc_cols,
        default=[loc_map.get("emp_type"), loc_map.get("emp_location")],
    )
    for col in sel_loc_cols:
        st.markdown(f"**Column:** `{col}`")
        existing = st.session_state["norm_loc_maps"].get(col, {})
        uniq = df_loc[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_loc_{col}")
        st.session_state["norm_loc_maps"][col] = {
            str(a): str(b) for a, b in edited.itertuples(index=False)
        }

with tab_ctrl:
    ctrl_cols = list(df_ctrl.columns)
    sel_ctrl_cols = st.multiselect(
        "Columns to normalize (Controlling)",
        options=ctrl_cols,
        default=[ctrl_map.get("emp_type_like"), ctrl_map.get("dept")],
    )
    for col in sel_ctrl_cols:
        st.markdown(f"**Column:** `{col}`")
        existing = st.session_state["norm_ctrl_maps"].get(col, {})
        uniq = df_ctrl[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_ctrl_{col}")
        st.session_state["norm_ctrl_maps"][col] = {
            str(a): str(b) for a, b in edited.itertuples(index=False)
        }

# =========================
# RUN
# =========================
st.markdown("---")
st.header("Run Disaggregation")

if st.button("Run processing", type="primary"):
    loc = normalize_columns(df_loc, st.session_state.get("norm_loc_maps", {}))
    ctrl = normalize_columns(df_ctrl, st.session_state.get("norm_ctrl_maps", {}))
    result = run_disagg(ctrl, loc, loc_map, ctrl_map)
    st.session_state["merged_result"] = result
    st.success("Processing completed successfully!")

# =========================
# RESULTS + PIVOT
# =========================
if "merged_result" in st.session_state:
    merged = st.session_state["merged_result"]

    st.subheader("Results snapshot")
    st.dataframe(merged.head(100), use_container_width=True)
# =========================
# PIVOT TABLE (Excel-style)
# =========================
st.markdown("---")
st.header("Pivot Table (Excel-like)")

if "merged_result" not in st.session_state:
    st.info("No data to pivot. Click **Run processing** first.")
else:
    merged = st.session_state["merged_result"]
    cols_all = list(merged.columns)

    # Default fields (auto-detected if exist)
    default_rows = [c for c in [ctrl_map.get("gb")] if c in cols_all]
    default_cols = [c for c in [ctrl_map.get("month")] if c in cols_all]
    default_filters = [c for c in [ctrl_map.get("dept"), ctrl_map.get("emp_type_like")] if c in cols_all]
    default_values = [v for v in ["Disagg_Value", "Budget_Disagg"] if v in cols_all]

    st.session_state.setdefault("pivot_rows", default_rows)
    st.session_state.setdefault("pivot_cols", default_cols)
    st.session_state.setdefault("pivot_filters", default_filters)
    st.session_state.setdefault("pivot_vals", default_values)
    st.session_state.setdefault("pivot_agg", "sum")

    with st.expander("Pivot configuration", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.selectbox("Aggregation", ["sum", "mean", "count"],
                         index=["sum","mean","count"].index(st.session_state["pivot_agg"]),
                         key="pivot_agg")
        with c2:
            st.number_input("Fill empty cells with", value=0.0, step=1.0, key="pivot_fill")

        c3, c4 = st.columns(2)
        with c3:
            st.multiselect("Rows", options=cols_all,
                           default=st.session_state["pivot_rows"],
                           key="pivot_rows")
        with c4:
            st.multiselect("Columns", options=cols_all,
                           default=st.session_state["pivot_cols"],
                           key="pivot_cols")

        st.markdown("**Filters**")
        st.multiselect("Filter fields", options=cols_all,
                       default=st.session_state["pivot_filters"],
                       key="pivot_filters")

        # Per-filter selectors (with "(All)")
        filter_value_keys = {}
        for fc in st.session_state["pivot_filters"]:
            uniq_vals = sorted(map(str, merged[fc].dropna().unique().tolist()))
            opt = ["(All)"] + uniq_vals
            key_name = f"flt_vals_{re.sub(r'[^A-Za-z0-9_]', '_', fc)}"
            filter_value_keys[fc] = key_name
            if key_name not in st.session_state:
                st.session_state[key_name] = ["(All)"]
            st.multiselect(f"{fc} values", options=opt,
                           default=st.session_state[key_name],
                           key=key_name)

    # Apply filters
    dfp = merged.copy()
    for fc in st.session_state["pivot_filters"]:
        sel = st.session_state.get(filter_value_keys.get(fc, ""), ["(All)"])
        if sel and "(All)" not in sel:
            dfp = dfp[dfp[fc].astype(str).isin(sel)]

    # Build pivot
    try:
        aggfunc = {"sum": np.sum, "mean": np.mean, "count": "count"}[st.session_state["pivot_agg"]]
        pivot_df = pd.pivot_table(
            dfp,
            index=st.session_state["pivot_rows"] or None,
            columns=st.session_state["pivot_cols"] or None,
            values=st.session_state["pivot_vals"] or None,
            aggfunc=aggfunc,
            fill_value=st.session_state["pivot_fill"],
            dropna=False,
        ).reset_index()

        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [" | ".join([str(x) for x in tup if x != ""]) for tup in pivot_df.columns.values]

        st.subheader("Pivot Result")
        st.dataframe(pivot_df, use_container_width=True)

        # Download pivot result
        pivot_buf = io.BytesIO()
        with pd.ExcelWriter(pivot_buf, engine="xlsxwriter") as writer:
            pivot_df.to_excel(writer, sheet_name="Pivot", index=False)
        st.download_button(
            "Download Pivot (Excel)",
            data=pivot_buf.getvalue(),
            file_name="pivot_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Pivot error: {e}")


    # Download result
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        merged.to_excel(writer, sheet_name="Result", index=False)
    st.download_button(
        "Download Excel result",
        data=out_buf.getvalue(),
        file_name="mapping_result.xlsx",
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
