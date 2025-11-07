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
# LAYER MAPPING (Dynamic)
# =========================
st.markdown("---")
st.header("Layer Mapping")

st.caption("Define multiple matching layers. Each layer must have 4 columns (e.g. GB, Dept, EmpType, Month).")

# --- Session init ---
st.session_state.setdefault("layers", [{"cols": [None, None, None, None]}])

def add_layer():
    st.session_state["layers"].append({"cols": [None, None, None, None]})
    st.rerun()

def remove_layer(i: int):
    if 0 <= i < len(st.session_state["layers"]):
        st.session_state["layers"].pop(i)
    st.rerun()

# --- Render each Layer ---
for i, layer in enumerate(st.session_state["layers"]):
    st.markdown(f"**Layer {i+1}** (select 4 columns)")
    cols = st.columns(4)
    loc_columns = list(df_loc.columns)
    for j, c in enumerate(cols):
        with c:
            opts = [None] + loc_columns
            selected = st.selectbox(
                f"Column {j+1}",
                options=opts,
                index=opts.index(layer["cols"][j]) if layer["cols"][j] in opts else 0,
                key=f"layer_{i}_col_{j}",
            )
            layer["cols"][j] = selected

    # Check if layer is valid
    valid = all(layer["cols"])
    msg = "✅ Ready" if valid else "⚠️ Please select all 4 columns"
    st.caption(msg)

    st.button("❌ Remove", key=f"remove_layer_{i}", on_click=remove_layer, args=(i,))

st.button("➕ Add Layer", on_click=add_layer)

# --- Save back ---
st.session_state["layers"] = st.session_state["layers"]

# --- Validation ---
all_valid = all(all(c for c in layer["cols"]) for layer in st.session_state["layers"])
if not all_valid:
    st.warning("Some layers are incomplete. Please select all 4 columns per layer.")
else:
    st.success(f"✅ {len(st.session_state['layers'])} layers configured properly!")

# =========================
# BREAK CONFIGURATION (separate section)
# =========================
st.markdown("---")
st.header("Break Configuration")

st.markdown(
    """
    <div style="padding:10px 16px; background-color:#f8f9fa; border-radius:8px; border-left:4px solid #007ACC;">
        <p style="margin:0; font-weight:500;">Select the dimension used for proportional breaking 
        (e.g. by <b>Location</b>, <b>Department</b>, <b>GB</b>, or <b>Employee Type</b>).</p>
    </div>
    """, unsafe_allow_html=True
)

break_candidates = ["emp_location", "dept", "gb", "emp_type"]
break_col_key = st.selectbox(
    "Select Break Dimension",
    options=break_candidates,
    index=0,
    key="break_dimension"
)

st.caption(f"→ The disaggregation will be computed based on **{break_col_key}** capacity.")


# =========================
# FLEXIBLE COLUMN MAPPING
# =========================
st.markdown("---")
st.header("Map Columns")

st.caption("Define how Location and Controlling datasets align.")

# ===== Join Key Mappings (safe add/remove with callbacks) =====
st.subheader("🔹 Join Key Mappings")

# state init
st.session_state.setdefault("join_mappings", [{"loc": None, "ctrl": None}])

def add_mapping():
    st.session_state["join_mappings"].append({"loc": None, "ctrl": None})
    st.rerun()

def remove_mapping(i: int):
    if 0 <= i < len(st.session_state["join_mappings"]):
        st.session_state["join_mappings"].pop(i)
    st.rerun()

join_maps = st.session_state["join_mappings"]

for i, pair in enumerate(join_maps):
    c1, c2, c3 = st.columns([3, 3, 1])

    with c1:
        loc_cols = [None] + list(df_loc.columns)
        idx = 0
        if pair.get("loc") in df_loc.columns:
            idx = loc_cols.index(pair["loc"])
        join_maps[i]["loc"] = st.selectbox(
            f"Location column {i+1}",
            options=loc_cols,
            index=idx,
            key=f"join_loc_{i}",
        )

    with c2:
        ctrl_cols = [None] + list(df_ctrl.columns)
        idx = 0
        if pair.get("ctrl") in df_ctrl.columns:
            idx = ctrl_cols.index(pair["ctrl"])
        join_maps[i]["ctrl"] = st.selectbox(
            f"Controlling column {i+1}",
            options=ctrl_cols,
            index=idx,
            key=f"join_ctrl_{i}",
        )

    with c3:
        st.button("❌", key=f"remove_map_{i}", on_click=remove_mapping, args=(i,))

st.button("➕ Add another mapping", on_click=add_mapping)


# --- SUPPORTING FIELDS SECTION ---
st.subheader("Supporting Fields (non-join columns)")

# Cột hỗ trợ ở Location
with st.expander("Location dataset"):
    support_loc = st.multiselect(
        "Select supporting columns (e.g. capacity, location, etc.)",
        options=list(df_loc.columns),
        default=[c for c in df_loc.columns if "capacity" in c.lower() or "location" in c.lower()],
        key="support_loc_cols"
    )

# Cột hỗ trợ ở Controlling
with st.expander("Controlling dataset"):
    support_ctrl = st.multiselect(
        "Select supporting columns (e.g. budget, rate, capacity, etc.)",
        options=list(df_ctrl.columns),
        default=[c for c in df_ctrl.columns if any(k in c.lower() for k in ["budget", "rate", "capacity"])],
        key="support_ctrl_cols"
    )

# Gộp lại thành mapping dictionary
loc_map = {"join_keys": [j["loc"] for j in join_maps if j["loc"]], "supporting": support_loc}
ctrl_map = {"join_keys": [j["ctrl"] for j in join_maps if j["ctrl"]], "supporting": support_ctrl}

st.session_state["loc_map"] = loc_map
st.session_state["ctrl_map"] = ctrl_map

# Hiển thị preview
st.markdown("**Current join pairs:**")
st.table(pd.DataFrame(join_maps))


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
    default_loc = [
        c for c in [loc_map.get("emp_type"), loc_map.get("emp_location")]
        if c in loc_cols
    ]
    sel_loc_cols = st.multiselect(
        "Columns to normalize (Location)",
        options=loc_cols,
        default=default_loc,
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
    default_ctrl = [
        c for c in [ctrl_map.get("emp_type_like"), ctrl_map.get("dept")]
        if c in ctrl_cols
    ]
    sel_ctrl_cols = st.multiselect(
        "Columns to normalize (Controlling)",
        options=ctrl_cols,
        default=default_ctrl,
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
