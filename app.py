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

st.markdown("""
This app performs **proportional disaggregation** of Controlling data 
based on capacity ratios from employee location.
""")

# =========================
# HELPERS
# =========================
def _clean_cols(df):
    df = df.copy()
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_excel(file):
    ext = Path(file.name).suffix.lower()
    engine = "openpyxl" if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"] else None
    xls = pd.ExcelFile(file, engine=engine)
    return {sn: _clean_cols(xls.parse(sn)) for sn in xls.sheet_names}

def normalize_columns(df, norm_maps):
    if not norm_maps:
        return df
    out = df.copy()
    for col, mapping in norm_maps.items():
        if col in out.columns and isinstance(mapping, dict):
            s = out[col].astype(str)
            out[col] = s.map(mapping).fillna(s)
    return out

def auto_match(left_df, right_df, cols_left, cols_right):
    """Tự động khớp cột giữa hai bảng kể cả khi tên khác nhau."""
    left_keys, right_keys = [], []
    for l, r in zip(cols_left, cols_right):
        if l in left_df.columns and r in right_df.columns:
            left_keys.append(l)
            right_keys.append(r)
        else:
            # fuzzy matching by cleaned name
            def find_match(col, pool):
                norm = lambda x: re.sub(r'[^a-z0-9]', '', str(x).lower())
                c_norm = norm(col)
                matches = [p for p in pool if norm(p) == c_norm]
                return matches[0] if matches else None
            l_match = find_match(l, left_df.columns)
            r_match = find_match(r, right_df.columns)
            if l_match and r_match:
                left_keys.append(l_match)
                right_keys.append(r_match)
    return left_keys, right_keys

def compute_ratio(df_loc, break_col, capacity_col, group_cols):
    df = df_loc.copy()
    grp_total = df.groupby(group_cols, dropna=False, as_index=False)[capacity_col].sum()
    grp_total = grp_total.rename(columns={capacity_col: "_cap_total"})
    grp_loc = df.groupby(group_cols + [break_col], dropna=False, as_index=False)[capacity_col].sum()
    grp_loc = grp_loc.rename(columns={capacity_col: "_cap_loc"})
    ratio_df = grp_loc.merge(grp_total, on=group_cols, how="left")
    ratio_df["Ratio"] = ratio_df["_cap_loc"] / ratio_df["_cap_total"].replace({0: np.nan})
    ratio_df["Ratio"] = ratio_df["Ratio"].fillna(0.0)
    return ratio_df

def run_disagg(ctrl, loc, mappings, layers, break_col):
    result = ctrl.copy()
    for idx, layer in enumerate(layers):
        cols = [c for c in layer["cols"] if c]
        if len(cols) < 1:
            continue
        st.write(f"Running layer {idx+1} with columns: {cols}")

        if break_col not in loc.columns:
            st.warning(f"Break column `{break_col}` not found in Location.")
            continue

        capacity_cols = [c for c in loc.columns if "capacity" in c.lower()]
        if not capacity_cols:
            st.warning("No capacity column found in Location.")
            continue

        cap_col = capacity_cols[0]
        ratio_df = compute_ratio(loc, break_col, cap_col, cols)

        left_cols, right_cols = auto_match(result, ratio_df, cols, cols)
        if not left_cols or not right_cols:
            st.warning(f"Layer {idx+1} skipped (no matching keys).")
            continue

        merged = result.merge(ratio_df, how="left", left_on=left_cols, right_on=right_cols)
        if "_cap_loc" not in merged.columns:
            merged["_cap_loc"] = 0
            merged["_cap_total"] = 1
            merged["Ratio"] = 0

        cap_ctrl = [c for c in ctrl.columns if "capacity" in c.lower()]
        budget_ctrl = [c for c in ctrl.columns if "budget" in c.lower()]
        rate_ctrl = [c for c in ctrl.columns if "rate" in c.lower()]

        cap_col_ctrl = cap_ctrl[0] if cap_ctrl else None
        bud_col_ctrl = budget_ctrl[0] if budget_ctrl else None
        rate_col_ctrl = rate_ctrl[0] if rate_ctrl else None

        if not all([cap_col_ctrl, bud_col_ctrl, rate_col_ctrl]):
            st.warning("Missing key columns (capacity/budget/rate) in Controlling.")
            continue

        merged["Disagg_Value"] = merged[cap_col_ctrl] * merged["Ratio"]
        share = merged["Disagg_Value"] / merged[cap_col_ctrl].replace({0: np.nan})
        share = np.nan_to_num(share, nan=0.0)
        merged["Budget_Disagg"] = merged[bud_col_ctrl] * share
        denom = (merged[rate_col_ctrl] * merged["Disagg_Value"]).replace({0: np.nan})
        merged["Billable_Disagg"] = merged[bud_col_ctrl] / denom
        merged["Billable_Disagg"] = merged["Billable_Disagg"].fillna(0.0)
        result = merged
    return result

# =========================
# SIDEBAR: FILE UPLOAD
# =========================
with st.sidebar:
    st.header("Upload Excel Files")
    file_loc = st.file_uploader("Upload Location file", type=["xlsx", "xlsm", "xls"])
    file_ctrl = st.file_uploader("Upload Controlling file", type=["xlsx", "xlsm", "xls"])
    if not file_loc or not file_ctrl:
        st.stop()

# Load and select sheets
loc_sheets = load_excel(file_loc)
ctrl_sheets = load_excel(file_ctrl)

sheet_loc = st.selectbox("Select Location Sheet", list(loc_sheets.keys()), index=0)
sheet_ctrl = st.selectbox("Select Controlling Sheet", list(ctrl_sheets.keys()), index=0)
df_loc = loc_sheets[sheet_loc]
df_ctrl = ctrl_sheets[sheet_ctrl]

# =========================
# BREAK CONFIG
# =========================
st.markdown("---")
st.header("Break Configuration")
break_col = st.selectbox("Select break dimension (e.g. Emp Location)", df_loc.columns, index=0)

# =========================
# MAPPING SECTION
# =========================
st.markdown("---")
st.header("Column Mapping")

st.session_state.setdefault("join_mappings", [{"loc": None, "ctrl": None}])
def add_mapping(): st.session_state["join_mappings"].append({"loc": None, "ctrl": None}); st.rerun()
def remove_mapping(i): st.session_state["join_mappings"].pop(i); st.rerun()

for i, mp in enumerate(st.session_state["join_mappings"]):
    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        st.session_state["join_mappings"][i]["loc"] = st.selectbox(
            f"Location column {i+1}", [None] + list(df_loc.columns), key=f"loc_map_{i}")
    with c2:
        st.session_state["join_mappings"][i]["ctrl"] = st.selectbox(
            f"Controlling column {i+1}", [None] + list(df_ctrl.columns), key=f"ctrl_map_{i}")
    with c3:
        st.button("Remove", key=f"remove_map_{i}", on_click=remove_mapping, args=(i,))
st.button("Add Mapping", on_click=add_mapping)

# =========================
# SUPPORTING FIELDS
# =========================
with st.expander("Supporting Fields"):
    st.session_state["support_loc"] = st.multiselect(
        "Location supporting fields", df_loc.columns)
    st.session_state["support_ctrl"] = st.multiselect(
        "Controlling supporting fields", df_ctrl.columns)

# =========================
# NORMALIZATION
# =========================
st.markdown("---")
st.header("Normalization Rules")
st.session_state.setdefault("norm_loc_maps", {})
st.session_state.setdefault("norm_ctrl_maps", {})

tab1, tab2 = st.tabs(["Location", "Controlling"])
with tab1:
    col_loc = st.multiselect("Normalize columns (Location)", df_loc.columns)
    for col in col_loc:
        uniq = df_loc[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"normloc_{col}")
        st.session_state["norm_loc_maps"][col] = {a: b for a, b in edited.itertuples(index=False)}

with tab2:
    col_ctrl = st.multiselect("Normalize columns (Controlling)", df_ctrl.columns)
    for col in col_ctrl:
        uniq = df_ctrl[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"normctrl_{col}")
        st.session_state["norm_ctrl_maps"][col] = {a: b for a, b in edited.itertuples(index=False)}

# =========================
# LAYER MAPPING
# =========================
st.markdown("---")
st.header("Layer Mapping")
st.caption("Each layer can have any number of columns.")

st.session_state.setdefault("layers", [{"cols": [None]}])
def add_layer(): st.session_state["layers"].append({"cols": [None]}); st.rerun()
def remove_layer(i): st.session_state["layers"].pop(i); st.rerun()

for i, layer in enumerate(st.session_state["layers"]):
    st.markdown(f"**Layer {i+1}**")
    num_cols = st.number_input(f"Number of columns in Layer {i+1}", 1, 10, len(layer["cols"]), key=f"num_{i}")
    if num_cols != len(layer["cols"]):
        layer["cols"] = (layer["cols"] + [None]*num_cols)[:num_cols]
    cols = st.columns(num_cols)
    for j, c in enumerate(cols):
        layer["cols"][j] = c.selectbox(f"Column {j+1}", [None] + list(df_loc.columns), key=f"layer_{i}_{j}")
    st.button("Remove Layer", key=f"remove_{i}", on_click=remove_layer, args=(i,))
st.button("Add Layer", on_click=add_layer)

# =========================
# RUN PROCESSING
# =========================
st.markdown("---")
st.header("Run Disaggregation")
if st.button("Run processing", type="primary"):
    loc = normalize_columns(df_loc, st.session_state["norm_loc_maps"])
    ctrl = normalize_columns(df_ctrl, st.session_state["norm_ctrl_maps"])
    result = run_disagg(ctrl, loc, st.session_state["join_mappings"], st.session_state["layers"], break_col)
    st.session_state["merged_result"] = result
    st.success("Processing completed successfully!")

# =========================
# PIVOT & GAP
# =========================
if "merged_result" in st.session_state:
    merged = st.session_state["merged_result"]
    st.subheader("Result Snapshot")
    st.dataframe(merged.head(100), use_container_width=True)

    st.markdown("---")
    st.header("Pivot Configuration")
    cols_all = merged.columns.tolist()
    rows = st.multiselect("Rows", cols_all)
    cols = st.multiselect("Columns", cols_all)
    vals = st.multiselect("Values", cols_all, default=["Disagg_Value", "Budget_Disagg"])
    aggfunc = st.selectbox("Aggregation", ["sum", "mean", "count"])
    pivot_df = pd.pivot_table(merged, index=rows, columns=cols, values=vals, aggfunc=aggfunc, fill_value=0)
    pivot_df = pivot_df.reset_index()
    st.dataframe(pivot_df, use_container_width=True)

    # Gap check vs original controlling
    st.markdown("---")
    st.header("Gap Check vs Controlling Original")
    try:
        ctrl_pivot = pd.pivot_table(df_ctrl, index=rows, columns=cols, values=vals, aggfunc=aggfunc, fill_value=0).reset_index()
        gap = pivot_df.set_index(rows) - ctrl_pivot.set_index(rows)
        gap = gap.fillna(0).reset_index()
        st.dataframe(gap, use_container_width=True)
        st.caption("Expect all 0 if disaggregation is correct.")
    except Exception as e:
        st.warning(f"Gap check skipped: {e}")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:13px; color:#666;'>Crafted with care by <strong>BGSV/CTG Data Team</strong>.</div>", unsafe_allow_html=True)
