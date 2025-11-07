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
# LAYER CONFIG (dynamic counts)
# =========================
st.markdown("---")
st.header("Layer Mapping")
st.caption("Set how many pairs are used per layer (must be <= number of valid join pairs). Example: if you have 4 pairs, Layer 1 = 4, Layer 2 = 3, Layer 3 = 1.")

valid_pairs = [p for p in join_maps if p.get("loc") and p.get("ctrl")]
max_pairs = len(valid_pairs)

st.session_state.setdefault("layer_sizes", [max_pairs] if max_pairs else [])

c_l, c_btn = st.columns([6, 1])
with c_l:
    # simple editor for a comma-separated list of integers
    default_text = ",".join(str(x) for x in st.session_state["layer_sizes"]) if st.session_state["layer_sizes"] else ""
    txt = st.text_input("Layer sizes (comma-separated)", value=default_text, key="layer_sizes_text")
with c_btn:
    if st.button("Apply"):
        parsed = []
        for tok in [t.strip() for t in txt.split(",") if t.strip()]:
            try:
                parsed.append(int(tok))
            except:
                pass
        st.session_state["layer_sizes"] = parsed
        st.rerun()

layer_sizes = [s for s in st.session_state.get("layer_sizes", []) if isinstance(s, int) and s > 0]
if max_pairs and not layer_sizes:
    st.info(f"Tip: You have {max_pairs} valid pairs. Try layer sizes like: {max_pairs},{max_pairs-1},1")
elif layer_sizes and any(s > max_pairs for s in layer_sizes):
    st.warning(f"Some layer sizes exceed number of valid pairs ({max_pairs}). They will be ignored during run.")

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

        result = run_disagg_layers(
            df_ctrl=ctrl_norm,
            df_loc=loc_norm,
            join_pairs=valid_pairs,
            layer_sizes=layer_sizes,
            loc_break_col=loc_break_col,
            loc_capacity_col=loc_capacity_col,
            ctrl_capacity_col=ctrl_capacity_col,
            ctrl_budget_col=ctrl_budget_col,
            ctrl_rate_col=ctrl_rate_col,
        )

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
