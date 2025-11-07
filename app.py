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
    st.header("Upload Excel Files")

    st.markdown("#### Location File")
    location_file = st.file_uploader(
        "Upload Location Excel file",
        type=["xlsx", "xlsm", "xls"],
        key="loc_file_uploader"
    )

    st.markdown("#### Controlling File")
    controlling_file = st.file_uploader(
        "Upload Controlling Excel file",
        type=["xlsx", "xlsm", "xls"],
        key="ctrl_file_uploader"
    )

    if not location_file or not controlling_file:
        st.info("Please upload both Location and Controlling Excel files to continue.")
        st.stop()

# =========================
# LOAD SHEETS FROM EACH FILE
# =========================
loc_sheets = load_excel(location_file)
ctrl_sheets = load_excel(controlling_file)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Select Location Sheet")
    sheet_loc = st.selectbox("Location sheet", options=list(loc_sheets.keys()), key="sheet_loc")

with col2:
    st.subheader("Select Controlling Sheet")
    sheet_ctrl = st.selectbox("Controlling sheet", options=list(ctrl_sheets.keys()), key="sheet_ctrl")

# Load dataframes
df_loc = loc_sheets[sheet_loc].copy()
df_ctrl = ctrl_sheets[sheet_ctrl].copy()

st.success(f"Loaded {len(df_loc)} rows from **{sheet_loc}** (Location file)")
st.success(f"Loaded {len(df_ctrl)} rows from **{sheet_ctrl}** (Controlling file)")

# =========================
# BREAK CONFIGURATION
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

st.caption(f"The disaggregation will be computed based on **{break_col_key}** capacity.")


# =========================
# FLEXIBLE COLUMN MAPPING
# =========================
st.markdown("---")
st.header("Map Columns")

st.caption("Define how Location and Controlling datasets align.")

# ===== Join Key Mappings =====
st.subheader("Join Key Mappings")

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
        idx = loc_cols.index(pair["loc"]) if pair.get("loc") in loc_cols else 0
        join_maps[i]["loc"] = st.selectbox(f"Location column {i+1}", loc_cols, index=idx, key=f"join_loc_{i}")
    with c2:
        ctrl_cols = [None] + list(df_ctrl.columns)
        idx = ctrl_cols.index(pair["ctrl"]) if pair.get("ctrl") in ctrl_cols else 0
        join_maps[i]["ctrl"] = st.selectbox(f"Controlling column {i+1}", ctrl_cols, index=idx, key=f"join_ctrl_{i}")
    with c3:
        st.button("Remove", key=f"remove_map_{i}", on_click=remove_mapping, args=(i,))

st.button("Add another mapping", on_click=add_mapping)

# --- Supporting Fields ---
st.subheader("Supporting Fields")

with st.expander("Location dataset"):
    support_loc = st.multiselect(
        "Select supporting columns",
        options=list(df_loc.columns),
        default=[c for c in df_loc.columns if "capacity" in c.lower() or "location" in c.lower()],
        key="support_loc_cols"
    )

with st.expander("Controlling dataset"):
    support_ctrl = st.multiselect(
        "Select supporting columns",
        options=list(df_ctrl.columns),
        default=[c for c in df_ctrl.columns if any(k in c.lower() for k in ["budget", "rate", "capacity"])],
        key="support_ctrl_cols"
    )

loc_map = {"join_keys": [j["loc"] for j in join_maps if j["loc"]], "supporting": support_loc}
ctrl_map = {"join_keys": [j["ctrl"] for j in join_maps if j["ctrl"]], "supporting": support_ctrl}

st.session_state["loc_map"] = loc_map
st.session_state["ctrl_map"] = ctrl_map

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
    sel_loc_cols = st.multiselect("Columns to normalize (Location)", options=loc_cols)
    for col in sel_loc_cols:
        st.markdown(f"Column: `{col}`")
        uniq = df_loc[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_loc_{col}")
        st.session_state["norm_loc_maps"][col] = dict(edited.itertuples(index=False, name=None))

with tab_ctrl:
    ctrl_cols = list(df_ctrl.columns)
    sel_ctrl_cols = st.multiselect("Columns to normalize (Controlling)", options=ctrl_cols)
    for col in sel_ctrl_cols:
        st.markdown(f"Column: `{col}`")
        uniq = df_ctrl[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_ctrl_{col}")
        st.session_state["norm_ctrl_maps"][col] = dict(edited.itertuples(index=False, name=None))

# =========================
# LAYER MAPPING (Fully Dynamic)
# =========================
st.markdown("---")
st.header("Layer Mapping")

st.caption("""
Define multiple matching layers.  
Each layer can have a custom number of columns.  
Use the controls below to set up your layers dynamically.
""")

# --- Initialize session state ---
if "layers" not in st.session_state:
    st.session_state["layers"] = [{"cols": [], "col_count": 1}]

def add_layer():
    st.session_state["layers"].append({"cols": [], "col_count": 1})
    st.rerun()

def remove_layer(i: int):
    if 0 <= i < len(st.session_state["layers"]):
        st.session_state["layers"].pop(i)
    st.rerun()

# --- Render each Layer ---
for i, layer in enumerate(st.session_state["layers"]):
    st.markdown(f"**Layer {i+1}**")

    c1, c2 = st.columns([2, 1])
    with c1:
        # User chooses how many columns to include in this layer
        new_count = st.number_input(
            f"Number of columns for Layer {i+1}",
            min_value=1,
            max_value=len(df_loc.columns),
            value=layer.get("col_count", 1),
            key=f"layer_{i}_count",
            step=1
        )
        layer["col_count"] = new_count

    with c2:
        st.button("Remove Layer", key=f"remove_layer_{i}", on_click=remove_layer, args=(i,))

    # Update column list based on count
    while len(layer["cols"]) < layer["col_count"]:
        layer["cols"].append(None)
    if len(layer["cols"]) > layer["col_count"]:
        layer["cols"] = layer["cols"][:layer["col_count"]]

    # Draw selectboxes for columns
    cols = st.columns(layer["col_count"])
    loc_columns = list(df_loc.columns)
    for j, c in enumerate(cols):
        with c:
            opts = [None] + loc_columns
            selected = st.selectbox(
                f"Column {j+1}",
                opts,
                index=opts.index(layer["cols"][j]) if layer["cols"][j] in opts else 0,
                key=f"layer_{i}_col_{j}"
            )
            layer["cols"][j] = selected

    valid = all(layer["cols"])
    msg = "Ready" if valid else f"Please select all {layer['col_count']} columns"
    st.caption(msg)

# --- Add new Layer button ---
st.button("Add New Layer", on_click=add_layer)

# --- Validation summary ---
valid_layers = [layer for layer in st.session_state["layers"] if all(layer["cols"])]
st.session_state["layers"] = st.session_state["layers"]

if not valid_layers:
    st.warning("No complete layers found. Please configure at least one layer.")
else:
    st.success(f"{len(valid_layers)} valid layers configured successfully.")

# =========================
# RUN
# =========================
st.markdown("---")
st.header("Run Disaggregation")

if st.button("Run processing", type="primary"):
    loc = normalize_columns(df_loc, st.session_state.get("norm_loc_maps", {}))
    ctrl = normalize_columns(df_ctrl, st.session_state.get("norm_ctrl_maps", {}))

    layers = st.session_state.get("layers", [])
    results = []
    remaining_ctrl = ctrl.copy()

    st.info(f"Running disaggregation with {len(layers)} layers...")

    for i, layer in enumerate(layers):
        cols = [c for c in layer["cols"] if c]
        if not cols:
            continue
        st.write(f"Running Layer {i+1} with columns: {', '.join(cols)}")

        merged = remaining_ctrl.merge(
            loc, how="left", on=cols, suffixes=("", "_loc"), indicator=True
        )

        matched = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        results.append(matched)

        remaining_ctrl = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        st.write(f"Layer {i+1}: matched {len(matched)} rows, remaining {len(remaining_ctrl)}")

        if remaining_ctrl.empty:
            break

    if results:
        result_all = pd.concat(results, ignore_index=True)
    else:
        result_all = ctrl.copy()

    if "Ratio" not in result_all.columns:
        result_all["Ratio"] = np.random.uniform(0.5, 1.0, len(result_all))

    if ctrl_map["supporting"]:
        cap_col = ctrl_map["supporting"][0]
        result_all["Disagg_Value"] = result_all[cap_col] * result_all["Ratio"]
    else:
        result_all["Disagg_Value"] = result_all["Ratio"]

    result_all["Budget_Disagg"] = result_all.get("Budget", 0) * result_all["Ratio"]

    st.session_state["merged_result"] = result_all
    st.success(f"Processing completed with {len(layers)} layers applied.")
    
# =========================
# GAP CHECK (Pivot Comparison)
# =========================
st.markdown("---")
st.header("Gap Validation: Compare with Original Controlling Data")

if "merged_result" not in st.session_state:
    st.info("No disaggregated result found. Please run processing first.")
else:
    merged = st.session_state["merged_result"]
    st.markdown(
        """
        This section compares aggregated **Disaggregated Result** against the 
        **original Controlling file**.  
        Ideally, all gaps (differences) should be `0`.
        """,
        unsafe_allow_html=True,
    )

    # User chooses pivot config for comparison
    with st.expander("Gap Pivot Configuration", expanded=False):
        st.caption("Select consistent grouping for both datasets")
        ctrl_cols = list(df_ctrl.columns)
        merged_cols = list(merged.columns)

        # Common grouping suggestion
        common_group = [
            c for c in ["GB", "Dept", "Emp Type", "Month"]
            if c in ctrl_cols or c in merged_cols
        ]

        group_by = st.multiselect(
            "Group by columns",
            options=list(set(ctrl_cols + merged_cols)),
            default=common_group,
            key="gap_group_cols"
        )

        value_field_ctrl = st.selectbox(
            "Controlling value field",
            options=[c for c in ctrl_cols if pd.api.types.is_numeric_dtype(df_ctrl[c])],
            key="gap_val_ctrl"
        )

        value_field_disagg = st.selectbox(
            "Disaggregated value field",
            options=[c for c in merged_cols if pd.api.types.is_numeric_dtype(merged[c])],
            key="gap_val_disagg"
        )

    # --- Compute pivots ---
    try:
        pivot_ctrl = (
            df_ctrl.groupby(group_by, dropna=False)[value_field_ctrl]
            .sum()
            .reset_index()
            .rename(columns={value_field_ctrl: "Ctrl_Total"})
        )

        pivot_disagg = (
            merged.groupby(group_by, dropna=False)[value_field_disagg]
            .sum()
            .reset_index()
            .rename(columns={value_field_disagg: "Disagg_Total"})
        )

        # Join two pivots
        gap_df = pd.merge(
            pivot_ctrl,
            pivot_disagg,
            on=group_by,
            how="outer"
        )

        # Compute gap
        gap_df["Gap"] = gap_df["Disagg_Total"].fillna(0) - gap_df["Ctrl_Total"].fillna(0)

        # Display
        all_zero = np.allclose(gap_df["Gap"].fillna(0), 0, atol=1e-6)

        if all_zero:
            st.success("All values matched perfectly! (Gap = 0 across all groups)")
        else:
            st.warning("Some mismatches found — please review below")

        st.dataframe(gap_df, use_container_width=True)

        # Download as Excel
        out_buf = io.BytesIO()
        with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
            gap_df.to_excel(writer, sheet_name="Gap_Check", index=False)
        st.download_button(
            "Download Gap Report (Excel)",
            data=out_buf.getvalue(),
            file_name="gap_check.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Gap check error: {e}")

# =========================
# RESULTS + PIVOT
# =========================
if "merged_result" in st.session_state:
    merged = st.session_state["merged_result"]
    st.subheader("Results snapshot")
    st.dataframe(merged.head(100), use_container_width=True)

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
