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

def guess_by_keywords(cols, keywords):
    cols_l = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(cols_l):
            if kw in c:
                return cols[i]
    return None

def compute_ratio_generic(df_loc, keys, break_col, cap_col):
    # keys: list of location columns (can be empty)
    use_cols = list(dict.fromkeys(keys + [break_col, cap_col]))
    df = df_loc[use_cols].copy()

    # sum by group + break
    cap_by = (
        df.groupby(keys + [break_col], dropna=False, as_index=False)[cap_col]
        .sum()
        .rename(columns={cap_col: "_cap_dim"})
    )
    # sum total by group
    if keys:
        cap_total = (
            df.groupby(keys, dropna=False, as_index=False)[cap_col]
            .sum()
            .rename(columns={cap_col: "_cap_total"})
        )
        ratio_df = cap_by.merge(cap_total, on=keys, how="left")
    else:
        total_val = df[cap_col].sum()
        cap_by["_cap_total"] = total_val
        ratio_df = cap_by

    ratio_df["Ratio"] = ratio_df["_cap_dim"] / ratio_df["_cap_total"].replace({0: np.nan})
    ratio_df["Ratio"] = ratio_df["Ratio"].fillna(0.0)
    return ratio_df  # contains keys + break_col + Ratio

def run_disagg_layers(
    df_ctrl, df_loc,
    join_pairs,             # list of {"loc": col_loc, "ctrl": col_ctrl}
    layer_sizes,            # e.g. [4, 3, 1]  (must be <= len(valid_pairs))
    loc_break_col,          # str in df_loc
    loc_capacity_col,       # str in df_loc
    ctrl_capacity_col,      # str in df_ctrl
    ctrl_budget_col,        # str in df_ctrl
    ctrl_rate_col           # str in df_ctrl
):
    # Validate pairs
    valid_pairs = [p for p in join_pairs if p.get("loc") and p.get("ctrl")]
    if not valid_pairs:
        raise ValueError("No valid join pairs set. Please select at least one Location/Controlling key pair.")

    max_pairs = len(valid_pairs)
    # sanitize layer sizes
    layer_sizes = [int(s) for s in layer_sizes if isinstance(s, int) and 0 < s <= max_pairs]
    if not layer_sizes:
        layer_sizes = [max_pairs]

    # start with all rows to process
    work = df_ctrl.copy()
    work["_matched"] = False
    work["_tmp_disagg"] = np.nan
    work["_tmp_ratio"] = np.nan

    # layer-by-layer
    merged_chunks = []

    for idx, L in enumerate(layer_sizes, start=1):
        # key lists for this layer
        keys_loc = [p["loc"] for p in valid_pairs[:L]]
        keys_ctrl = [p["ctrl"] for p in valid_pairs[:L]]

        # compute ratio on location by keys + break_col
        ratio_df = compute_ratio_generic(df_loc, keys_loc, break_col=loc_break_col, cap_col=loc_capacity_col)

        # take only rows not yet matched
        chunk = work.loc[~work["_matched"]].copy()

        # if any key missing in either side -> skip this layer
        if not all(k in chunk.columns for k in keys_ctrl) or not all(k in ratio_df.columns for k in keys_loc):
            continue

        # align column names for merge: build left_on/right_on
        left_on = keys_ctrl
        right_on = keys_loc

        # merge and compute temporary disagg_value
        merged = chunk.merge(
            ratio_df,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffixes=("", "_r")
        )

        # Disagg base value proportional to ctrl capacity
        cap_series = merged[ctrl_capacity_col].replace({0: np.nan})
        merged["_tmp_disagg"] = merged[ctrl_capacity_col] * merged["Ratio"]
        share = merged["_tmp_disagg"] / cap_series
        share = np.nan_to_num(share, nan=0.0)

        merged["_tmp_budget"] = merged[ctrl_budget_col] * share
        denom = (merged[ctrl_rate_col] * merged["_tmp_disagg"]).replace({0: np.nan})
        merged["_tmp_billable"] = merged[ctrl_budget_col] / denom
        merged["_tmp_billable"] = merged["_tmp_billable"].fillna(0.0)

        # rows that got a ratio (matched) are marked; keep others for next layer
        got_ratio = merged["Ratio"].notna()
        merged["_matched"] = got_ratio
        merged_chunks.append(merged)

        # update main work table matching flags by index
        work.loc[merged.index, "_matched"] = got_ratio
        work.loc[merged.index, "_tmp_disagg"] = merged["_tmp_disagg"]
        work.loc[merged.index, "_tmp_ratio"] = merged["Ratio"]

    # concat all processed pieces; include unmatched rows (Ratio NaN -> 0)
    if merged_chunks:
        out = pd.concat(merged_chunks, ignore_index=True)
    else:
        out = work.copy()

    out["Ratio"] = out["_tmp_ratio"].fillna(0.0)
    out["Disagg_Value"] = out["_tmp_disagg"].fillna(0.0)
    out["Budget_Disagg"] = np.where(
        out["Disagg_Value"].eq(0),
        0.0,
        out["_tmp_budget"].fillna(0.0) if "_tmp_budget" in out.columns else 0.0
    )
    out["Billable_Disagg"] = np.where(
        out["Disagg_Value"].eq(0),
        0.0,
        out["_tmp_billable"].fillna(0.0) if "_tmp_billable" in out.columns else 0.0
    )

    # cleanup temp cols
    drop_cols = [c for c in ["_tmp_disagg", "_tmp_ratio", "_tmp_budget", "_tmp_billable"] if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")

    return out

def normalize_cols_simple(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# =========================
# SIDEBAR: UPLOAD TWO FILES
# =========================
with st.sidebar:
    st.header("Upload Excel files")
    file_loc = st.file_uploader("Location Excel file", type=["xlsx", "xlsm", "xls"], key="file_loc")
    file_ctrl = st.file_uploader("Controlling Excel file", type=["xlsx", "xlsm", "xls"], key="file_ctrl")

    if not file_loc or not file_ctrl:
        st.info("Upload both Location and Controlling Excel files to continue.")
        st.stop()

# =========================
# LOAD SHEETS (two files)
# =========================
sheets_loc = load_excel(file_loc)
sheets_ctrl = load_excel(file_ctrl)

sheet_loc = st.selectbox("Location sheet", options=list(sheets_loc.keys()), index=0, key="sheet_loc")
sheet_ctrl = st.selectbox("Controlling sheet", options=list(sheets_ctrl.keys()), index=0, key="sheet_ctrl")

df_loc = normalize_cols_simple(sheets_loc[sheet_loc].copy())
df_ctrl = normalize_cols_simple(sheets_ctrl[sheet_ctrl].copy())

# =========================
# BREAK + REQUIRED FIELDS
# =========================
st.markdown("---")
st.header("Break & Required Fields")

# break column (in LOCATION)
default_break_guess = guess_by_keywords(df_loc.columns, ["location", "site", "emp location"])
loc_break_col = st.selectbox(
    "Break column in Location (dimension you want to disaggregate by)",
    options=list(df_loc.columns),
    index=list(df_loc.columns).index(default_break_guess) if default_break_guess in df_loc.columns else 0,
    key="loc_break_col"
)

# capacity in LOCATION
default_loc_cap = guess_by_keywords(df_loc.columns, ["capacity"])
loc_capacity_col = st.selectbox(
    "Location Capacity column",
    options=list(df_loc.columns),
    index=list(df_loc.columns).index(default_loc_cap) if default_loc_cap in df_loc.columns else 0,
    key="loc_capacity_col"
)

# required numeric fields in CONTROLLING
default_ctrl_cap = guess_by_keywords(df_ctrl.columns, ["capacity"])
default_ctrl_budget = guess_by_keywords(df_ctrl.columns, ["budget"])
default_ctrl_rate = guess_by_keywords(df_ctrl.columns, ["rate", "selling"])

ctrl_capacity_col = st.selectbox(
    "Controlling Capacity column",
    options=list(df_ctrl.columns),
    index=list(df_ctrl.columns).index(default_ctrl_cap) if default_ctrl_cap in df_ctrl.columns else 0,
    key="ctrl_capacity_col"
)
ctrl_budget_col = st.selectbox(
    "Controlling Budget column",
    options=list(df_ctrl.columns),
    index=list(df_ctrl.columns).index(default_ctrl_budget) if default_ctrl_budget in df_ctrl.columns else 0,
    key="ctrl_budget_col"
)
ctrl_rate_col = st.selectbox(
    "Controlling Rate column",
    options=list(df_ctrl.columns),
    index=list(df_ctrl.columns).index(default_ctrl_rate) if default_ctrl_rate in df_ctrl.columns else 0,
    key="ctrl_rate_col"
)

# =========================
# FLEXIBLE JOIN PAIRS
# =========================
st.markdown("---")
st.header("Join Key Mappings")
st.caption("Define aligned key pairs used for joining Location and Controlling. Top N pairs are used for Layer 1; fewer pairs for lower layers.")

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
    c1, c2, c3 = st.columns([4, 4, 1])
    with c1:
        loc_cols = [None] + list(df_loc.columns)
        idx = loc_cols.index(pair["loc"]) if pair.get("loc") in loc_cols else 0
        join_maps[i]["loc"] = st.selectbox(
            f"Location column {i+1}",
            options=loc_cols,
            index=idx,
            key=f"join_loc_{i}",
        )
    with c2:
        ctrl_cols = [None] + list(df_ctrl.columns)
        idx = ctrl_cols.index(pair["ctrl"]) if pair.get("ctrl") in ctrl_cols else 0
        join_maps[i]["ctrl"] = st.selectbox(
            f"Controlling column {i+1}",
            options=ctrl_cols,
            index=idx,
            key=f"join_ctrl_{i}",
        )
    with c3:
        st.button("Remove", key=f"remove_map_{i}", on_click=remove_mapping, args=(i,))

st.button("Add another mapping", on_click=add_mapping)

# =========================
# =========================
# LAYER MAPPING (Dynamic + Adjustable Columns)
# =========================
st.markdown("---")
st.header("Layer Mapping")

st.caption("""
Define multiple matching layers dynamically.  
Each layer can have a custom number of columns (1–6 for example).  
Higher layers use more detailed keys; lower layers use fewer keys.
""")

# --- Session init ---
st.session_state.setdefault("layers", [{"n_cols": 4, "cols": [None]*4}])

def add_layer():
    st.session_state["layers"].append({"n_cols": 4, "cols": [None]*4})
    st.rerun()

def remove_layer(i: int):
    if 0 <= i < len(st.session_state["layers"]):
        st.session_state["layers"].pop(i)
    st.rerun()

# --- Render each Layer ---
for i, layer in enumerate(st.session_state["layers"]):
    st.markdown(f"**Layer {i+1}**")

    # Select number of columns
    c_num, _ = st.columns([1, 6])
    with c_num:
        n_cols = st.number_input(
            "Number of columns",
            min_value=1, max_value=10,
            value=layer.get("n_cols", 4),
            step=1,
            key=f"ncols_{i}"
        )
        if n_cols != layer.get("n_cols"):
            layer["n_cols"] = n_cols
            layer["cols"] = (layer["cols"] + [None]*n_cols)[:n_cols]

    # Render selectboxes for that layer
    st.caption(f"Select {layer['n_cols']} columns for Layer {i+1}")
    cols_row = st.columns(layer["n_cols"])
    loc_columns = list(df_loc.columns)

    for j, c in enumerate(cols_row):
        with c:
            opts = [None] + loc_columns
            selected = st.selectbox(
                f"Column {j+1}",
                options=opts,
                index=opts.index(layer["cols"][j]) if layer["cols"][j] in opts else 0,
                key=f"layer_{i}_col_{j}"
            )
            layer["cols"][j] = selected

    valid = all(layer["cols"])
    msg = "Ready" if valid else "Please select all columns"
    st.caption(msg)

    st.button("Remove Layer", key=f"remove_layer_{i}", on_click=remove_layer, args=(i,))

st.button("Add Layer", on_click=add_layer)

# --- Validation summary ---
st.session_state["layers"] = st.session_state["layers"]
all_valid = all(all(c for c in layer["cols"]) for layer in st.session_state["layers"])
if not all_valid:
    st.warning("Some layers are incomplete. Please select all columns per layer.")
else:
    st.success(f"{len(st.session_state['layers'])} layers configured properly!")

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
        default=[],
    )
    for col in sel_loc_cols:
        st.markdown(f"**Column:** `{col}`")
        uniq = df_loc[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_loc_{col}")
        st.session_state["norm_loc_maps"][col] = {str(a): str(b) for a, b in edited.itertuples(index=False)}

with tab_ctrl:
    ctrl_cols = list(df_ctrl.columns)
    sel_ctrl_cols = st.multiselect(
        "Columns to normalize (Controlling)",
        options=ctrl_cols,
        default=[],
    )
    for col in sel_ctrl_cols:
        st.markdown(f"**Column:** `{col}`")
        uniq = df_ctrl[col].dropna().astype(str).unique().tolist()
        edit_df = pd.DataFrame({"from": uniq, "to": uniq})
        edited = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_ctrl_{col}")
        st.session_state["norm_ctrl_maps"][col] = {str(a): str(b) for a, b in edited.itertuples(index=False)}

# =========================
# RUN
# =========================
st.markdown("---")
st.header("Run Disaggregation")

if st.button("Run processing", type="primary"):
    try:
        loc_norm = normalize_columns(df_loc, st.session_state.get("norm_loc_maps", {}))
        ctrl_norm = normalize_columns(df_ctrl, st.session_state.get("norm_ctrl_maps", {}))

        # --- Extract layer info ---
        layer_config = st.session_state.get("layers", [])
        if not layer_config:
            st.error("Please define at least one Layer before running.")
            st.stop()

        # --- Run layer-by-layer disaggregation ---
        merged_chunks = []
        remaining_ctrl = ctrl_norm.copy()

        for idx, layer in enumerate(layer_config, start=1):
            cols = [c for c in layer["cols"] if c]
            if not cols:
                continue

            st.write(f"Running Layer {idx} with {len(cols)} columns: {cols}")

            # Compute ratio by these columns + break column
            ratio_df = compute_ratio_generic(
                df_loc=loc_norm,
                keys=cols,
                break_col=loc_break_col,
                cap_col=loc_capacity_col,
            )

            # Merge & compute disagg
            # --- Build safe matching keys (auto align by name, case & space-insensitive) ---
            left_cols, right_cols = [], []
            for c in cols:
                # tìm cột bên trái
                l_match = next((l for l in remaining_ctrl.columns if l.lower().replace(" ", "") == c.lower().replace(" ", "")), None)
                r_match = next((r for r in ratio_df.columns if r.lower().replace(" ", "") == c.lower().replace(" ", "")), None)
                if l_match and r_match:
                    left_cols.append(l_match)
                    right_cols.append(r_match)
            
            # --- Nếu không có cột trùng thì bỏ qua layer ---
            if not left_cols or not right_cols:
                st.warning(f"Layer {idx} skipped — no common keys between datasets.")
                continue
            
            # --- Bảo đảm có cùng độ dài ---
            if len(left_cols) != len(right_cols):
                min_len = min(len(left_cols), len(right_cols))
                left_cols = left_cols[:min_len]
                right_cols = right_cols[:min_len]
            
            # --- Merge an toàn ---
            merged = remaining_ctrl.merge(
                ratio_df,
                how="left",
                left_on=left_cols,
                right_on=right_cols,
                suffixes=("", "_r"),
            )



            # Base disaggregation
            merged["Disagg_Value"] = merged[ctrl_capacity_col] * merged["Ratio"].fillna(0)
            share = np.where(
                merged[ctrl_capacity_col] == 0, 0, merged["Disagg_Value"] / merged[ctrl_capacity_col]
            )
            merged["Budget_Disagg"] = merged[ctrl_budget_col] * share
            denom = (merged[ctrl_rate_col] * merged["Disagg_Value"]).replace({0: np.nan})
            merged["Billable_Disagg"] = np.where(
                merged["Disagg_Value"].eq(0), 0, merged[ctrl_budget_col] / denom
            )

            # Rows matched in this layer (Ratio not null)
            matched = merged["Ratio"].notna()
            merged_chunks.append(merged.loc[matched])
            remaining_ctrl = merged.loc[~matched].drop(columns=["Ratio"], errors="ignore")

        # Combine all results
        if merged_chunks:
            result = pd.concat(merged_chunks + [remaining_ctrl], ignore_index=True)
        else:
            result = ctrl_norm.copy()

        result["Disagg_Value"] = result.get("Disagg_Value", 0).fillna(0)
        result["Budget_Disagg"] = result.get("Budget_Disagg", 0).fillna(0)
        result["Billable_Disagg"] = result.get("Billable_Disagg", 0).fillna(0)

        st.session_state["merged_result"] = result
        st.success("Processing completed successfully!")

    except Exception as e:
        st.error(f"Run error: {e}")

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
    default_rows = []
    default_cols = []
    default_filters = []
    default_values = [v for v in ["Disagg_Value", "Budget_Disagg"] if v in cols_all]

    st.session_state.setdefault("pivot_rows", default_rows)
    st.session_state.setdefault("pivot_cols", default_cols)
    st.session_state.setdefault("pivot_filters", default_filters)
    st.session_state.setdefault("pivot_vals", default_values)
    st.session_state.setdefault("pivot_agg", "sum")

    with st.expander("Pivot configuration", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.selectbox(
                "Aggregation", ["sum", "mean", "count"],
                index=["sum","mean","count"].index(st.session_state["pivot_agg"]),
                key="pivot_agg"
            )
        with c2:
            st.number_input("Fill empty cells with", value=0.0, step=1.0, key="pivot_fill")

        c3, c4 = st.columns(2)
        with c3:
            st.multiselect("Rows", options=cols_all, default=st.session_state["pivot_rows"], key="pivot_rows")
        with c4:
            st.multiselect("Columns", options=cols_all, default=st.session_state["pivot_cols"], key="pivot_cols")

        st.markdown("**Filters**")
        st.multiselect("Filter fields", options=cols_all, default=st.session_state["pivot_filters"], key="pivot_filters")

        # Per-filter selectors (with "(All)")
        filter_value_keys = {}
        for fc in st.session_state["pivot_filters"]:
            uniq_vals = sorted(map(str, merged[fc].dropna().unique().tolist()))
            opt = ["(All)"] + uniq_vals
            key_name = f"flt_vals_{re.sub(r'[^A-Za-z0-9_]', '_', fc)}"
            filter_value_keys[fc] = key_name
            if key_name not in st.session_state:
                st.session_state[key_name] = ["(All)"]
            st.multiselect(f"{fc} values", options=opt, default=st.session_state[key_name], key=key_name)

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

    # Download raw result
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        merged.to_excel(writer, sheet_name="Result", index=False)
    st.download_button(
        "Download Excel result",
        data=out_buf.getvalue(),
        file_name="disagg_result.xlsx",
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
