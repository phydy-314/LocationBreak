import io
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Stimulation - no more pain :))", layout="wide")
st.title("Stimulation - no more pain :))")

# =========================
# CONSTANTS
# =========================
DEFAULT_NORMALIZATION = {
    "emp_type_map": {
        "Others-Internal": "Offshore",
        "Offshore through ICT": "Offshore - through ICT",
        "Outsourcing through ICT Capacity": "Outsourcing - through ICT",
    },
    "emp_location_map": {
        "HO CHI MINH": "HCMC",
    },
}

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
    """Trim and normalize weird spaces in column names."""
    df = df.copy()
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name=None):
    """Load Excel with proper engine; clean column names; show clear error if engine is missing."""
    name = getattr(file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"          # xlrd==1.2.0 is needed for .xls
    elif ext == ".xlsb":
        engine = "pyxlsb"        # pyxlsb is needed for .xlsb
    else:
        engine = None

    try:
        xls = pd.ExcelFile(file, engine=engine)
    except ImportError:
        st.error(
            "Missing Excel engine for this format. Add to requirements.txt:\n"
            "- openpyxl (xlsx/xlsm)\n- xlrd==1.2.0 (xls)\n- pyxlsb (xlsb)"
        )
        raise

    if sheet_name is None:
        return {sn: _clean_cols(xls.parse(sn)) for sn in xls.sheet_names}
    else:
        return {sheet_name: _clean_cols(xls.parse(sheet_name))}

def normalize_columns(df: pd.DataFrame, norm_maps: dict) -> pd.DataFrame:
    """
    Apply normalization for multiple columns.
    norm_maps = { "col_name": {"from1":"to1", ...}, ... }
    """
    if not norm_maps:
        return df
    out = df.copy()
    for col, mapping in norm_maps.items():
        if col in out.columns and isinstance(mapping, dict) and mapping:
            s = out[col].astype(str)
            out[col] = s.map(mapping).fillna(s)
    return out

def sample_mapping_table(df: pd.DataFrame, col: str, existing: dict | None, max_unique: int = 150) -> pd.DataFrame:
    """
    Provide a 2-col [from, to] table for data_editor.
    Prefill from existing map, or build identity pairs from distinct values.
    """
    if existing:
        rows = [(k, v) for k, v in existing.items()]
        return pd.DataFrame(rows, columns=["from", "to"])
    if col not in df.columns:
        return pd.DataFrame(columns=["from", "to"])
    uniq = (
        df[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    uniq = uniq[:max_unique]
    return pd.DataFrame({"from": uniq, "to": uniq})

def compute_ratio_expanding(df_loc, keys, emp_loc_col, cap_col):
    """Build location split ratios by KEYS + Emp Location. If total=0 → Ratio=0 (still treated as matched)."""
    df = df_loc.copy()
    use_cols = [c for c in keys + [emp_loc_col, cap_col] if c in df.columns]
    df = df[use_cols].copy()

    cap_by_loc = (
        df.groupby(keys + [emp_loc_col], dropna=False, as_index=False)[cap_col]
        .sum().rename(columns={cap_col: "_cap_loc"})
    )
    cap_total = (
        df.groupby(keys, dropna=False, as_index=False)[cap_col]
        .sum().rename(columns={cap_col: "_cap_total"})
    )
    ratio_df = cap_by_loc.merge(cap_total, on=keys, how="left")
    denom = ratio_df["_cap_total"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_df["Ratio"] = ratio_df["_cap_loc"] / denom.replace({0: np.nan})
    ratio_df.loc[denom.fillna(0).eq(0), "Ratio"] = 0.0
    return ratio_df.drop(columns=["_cap_loc", "_cap_total"])

def coalesce_to(df: pd.DataFrame, canonical: str, base_names: list[str]) -> pd.DataFrame:
    """Coalesce column variants (base + *_x/_y/...) into one `canonical` column."""
    cand_cols = []
    for base in base_names:
        if base in df.columns:
            cand_cols.append(base)
        for c in df.columns:
            if c != base and str(c).startswith(base + "_"):
                cand_cols.append(c)
    seen = set()
    ordered = [c for c in cand_cols if not (c in seen or seen.add(c))]
    if not ordered:
        return df
    merged = df[ordered].bfill(axis=1).iloc[:, 0]
    df[canonical] = merged
    drop_cols = [c for c in ordered if c != canonical]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df

def run_fallback_merges_expand(df_ctrl, df_loc, m_loc, m_ctrl, enabled_layers):
    """
    Expand-by-location across 4 layers with staged append:
      - At each layer: take only unmapped rows, expand by location, append mapped rows back.
      - Unmapped after L4 are not appended.
      - Coalesce potential *_x/_y suffixes after merges.
    """
    base = df_ctrl.copy()
    if "Capacity_Location" not in base.columns:
        base["Capacity_Location"] = np.nan
    if "MappedLayer" not in base.columns:
        base["MappedLayer"] = pd.Series([None] * len(base), dtype="object")

    emp_loc_col = m_loc["emp_location"]
    merge_info = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}

    def _layer(base_df, layer_keys_loc, layer_tag):
        mask = base_df["Capacity_Location"].isna()
        if mask.sum() == 0:
            return base_df, pd.DataFrame(), 0

        work = base_df.loc[mask].copy()
        base_df = base_df.loc[~mask].copy()

        ratio_df = compute_ratio_expanding(df_loc, layer_keys_loc, emp_loc_col, m_loc["capacity_loc"])

        if layer_tag == "L1":
            left_keys = [m_ctrl["gb"], m_ctrl["dept"], m_ctrl["emp_type_like"], m_ctrl["month"]]
            right_keys = [m_loc["gb"],  m_loc["dept"],  m_loc["emp_type"],       m_loc["month"]]
        elif layer_tag == "L2":
            left_keys = [m_ctrl["dept"], m_ctrl["emp_type_like"], m_ctrl["month"]]
            right_keys = [m_loc["dept"],  m_loc["emp_type"],       m_loc["month"]]
        elif layer_tag == "L3":
            left_keys = [m_ctrl["gb"], m_ctrl["dept"], m_ctrl["month"]]
            right_keys = [m_loc["gb"],  m_loc["dept"],  m_loc["month"]]
        else:  # L4
            left_keys = [m_ctrl["dept"], m_ctrl["month"]]
            right_keys = [m_loc["dept"],  m_loc["month"]]

        drop_list = [c for c in work.columns if c in ("Ratio", "Capacity_Location_tmp")
                     or str(c).endswith("_x") or str(c).endswith("_y")]
        work = work.drop(columns=drop_list, errors="ignore")

        work_merged = work.merge(ratio_df, left_on=left_keys, right_on=right_keys, how="left")

        for (canonical, bases) in [
            (m_ctrl["gb"],            [m_ctrl["gb"], m_loc["gb"]]),
            (m_ctrl["dept"],          [m_ctrl["dept"], m_loc["dept"]]),
            (m_ctrl["emp_type_like"], [m_ctrl["emp_type_like"], m_loc["emp_type"]]),
            (m_ctrl["month"],         [m_ctrl["month"], m_loc["month"]]),
            (emp_loc_col,             [emp_loc_col]),
        ]:
            work_merged = coalesce_to(work_merged, canonical=canonical, base_names=bases)

        work_merged["Capacity_Location_tmp"] = work_merged[m_ctrl["capacity"]] * work_merged["Ratio"]
        matched_mask = work_merged["Ratio"].notna()
        matched   = work_merged.loc[matched_mask].copy()
        unmatched = work_merged.loc[~matched_mask].copy()

        matched["Capacity_Location"] = matched["Capacity_Location_tmp"]
        matched["MappedLayer"] = layer_tag

        if emp_loc_col not in base_df.columns:
            base_df[emp_loc_col] = np.nan
        matched_aligned = matched.reindex(columns=base_df.columns, fill_value=np.nan)
        base_df = pd.concat([base_df, matched_aligned], ignore_index=True)
        return base_df, unmatched, int(matched_mask.sum())

    keys_L1_loc = [m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]]
    keys_L2_loc = [m_loc["dept"], m_loc["emp_type"], m_loc["month"]]
    keys_L3_loc = [m_loc["gb"], m_loc["dept"], m_loc["month"]]
    keys_L4_loc = [m_loc["dept"], m_loc["month"]]

    work_next = pd.DataFrame()
    if enabled_layers.get("L1", True):
        base, work_next, cnt = _layer(base, keys_L1_loc, "L1"); merge_info["L1"] = cnt
    else:
        work_next = base.loc[base["Capacity_Location"].isna()].copy()

    if enabled_layers.get("L2", True) and not work_next.empty:
        base = pd.concat([base, work_next], ignore_index=True)
        base, work_next, cnt = _layer(base, keys_L2_loc, "L2"); merge_info["L2"] = cnt

    if enabled_layers.get("L3", True) and not work_next.empty:
        base = pd.concat([base, work_next], ignore_index=True)
        base, work_next, cnt = _layer(base, keys_L3_loc, "L3"); merge_info["L3"] = cnt

    if enabled_layers.get("L4", True) and not work_next.empty:
        base = pd.concat([base, work_next], ignore_index=True)
        base, work_next, cnt = _layer(base, keys_L4_loc, "L4"); merge_info["L4"] = cnt

    return base, merge_info

def compute_outputs(df, m_ctrl, on_div0="zero"):
    df = df.copy()
    cap_series = df[m_ctrl["capacity"]]
    denom_cap = cap_series.replace({0: np.nan})
    share = df["Capacity_Location"] / denom_cap
    share = np.nan_to_num(share, nan=0.0)

    df["Budget Location"] = df[m_ctrl["budget"]] * share

    denom = (df[m_ctrl["rate"]] * df.get("Capacity_Location", 0)).replace({0: np.nan})
    billable = df[m_ctrl["budget"]] / denom
    if on_div0 == "zero":
        billable = billable.fillna(0.0)
    elif on_div0 == "blank":
        billable = billable.where(billable.notna(), None)
    else:
        billable = billable.fillna(0.0)

    df["Billable Capacity Location"] = billable
    return df

def guess(colnames, candidates):
    col_lower = {str(c).lower(): c for c in colnames}
    for c in candidates:
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None

def ensure_month_order(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """Force month column to ordered categorical (1..12) while keeping original labels."""
    if month_col not in df.columns:
        return df.copy()

    df = df.copy()
    m = {
        "jan": 1, "january": 1, "01": 1, "1": 1,
        "feb": 2, "february": 2, "02": 2, "2": 2,
        "mar": 3, "march": 3, "03": 3, "3": 3,
        "apr": 4, "april": 4, "04": 4, "4": 4,
        "may": 5, "05": 5, "5": 5,
        "jun": 6, "june": 6, "06": 6, "6": 6,
        "jul": 7, "july": 7, "07": 7, "7": 7,
        "aug": 8, "august": 8, "08": 8, "8": 8,
        "sep": 9, "sept": 9, "september": 9, "09": 9, "9": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    def month_num(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        try:
            n = int(s)
            if 1 <= n <= 12: return n
        except: pass
        return m.get(s.lower(), np.nan)

    pairs = []
    for v in df[month_col].dropna().unique().tolist():
        n = month_num(v)
        if not np.isnan(n): pairs.append((v, int(n)))
    if not pairs: return df

    pairs_sorted = sorted(pairs, key=lambda t: t[1])
    ordered_labels = [v for v,_ in pairs_sorted]
    df[month_col] = pd.Categorical(df[month_col], categories=ordered_labels, ordered=True)
    return df

def keysafe(s: str) -> str:
    """Sanitize widget keys from arbitrary column names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s))

def validate_mapping(df_loc, df_ctrl, m_loc, m_ctrl):
    required_loc = ["gb","dept","emp_type","month","emp_location","capacity_loc"]
    required_ctrl = ["gb","dept","emp_type_like","month","capacity","budget","rate"]
    missing = {"Location": [], "Controlling": []}
    for k in required_loc:
        col = m_loc.get(k)
        if not col or col not in df_loc.columns:
            missing["Location"].append((k, col))
    for k in required_ctrl:
        col = m_ctrl.get(k)
        if not col or col not in df_ctrl.columns:
            missing["Controlling"].append((k, col))
    return missing

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Upload")
    excel_file = st.file_uploader("Upload .xlsx/.xlsm/.xls/.xlsb", type=["xlsx", "xlsm", "xls", "xlsb"])
    on_div0 = st.selectbox("Divide-by-zero handling", ["zero", "blank"], index=0)

if not excel_file:
    st.info("Upload your Excel to get started.")
    st.stop()

# =========================
# LOAD SHEETS
# =========================
all_sheets = load_excel(excel_file)

st.header("Select Sheets")
cols = st.columns(2)
with cols[0]:
    sheet_loc = st.selectbox(
        "Location sheet",
        options=list(all_sheets.keys()),
        index=(list(all_sheets.keys()).index("Location") if "Location" in all_sheets else 0),
    )
with cols[1]:
    sheet_ctrl = st.selectbox(
        "Controlling sheet",
        options=list(all_sheets.keys()),
        index=(list(all_sheets.keys()).index("Controlling") if "Controlling" in all_sheets else 0),
    )

df_loc = all_sheets[sheet_loc].copy()
df_ctrl = all_sheets[sheet_ctrl].copy()

# Reset state when sheets change
if "last_sheet_loc" not in st.session_state:
    st.session_state["last_sheet_loc"] = sheet_loc
if "last_sheet_ctrl" not in st.session_state:
    st.session_state["last_sheet_ctrl"] = sheet_ctrl

sheet_changed = (
    st.session_state["last_sheet_loc"] != sheet_loc
    or st.session_state["last_sheet_ctrl"] != sheet_ctrl
)
if sheet_changed:
    for k in list(st.session_state.keys()):
        if k.startswith("loc_") or k.startswith("ctrl_"):
            del st.session_state[k]
    st.session_state.pop("norm_loc_maps", None)
    st.session_state.pop("norm_ctrl_maps", None)
    st.session_state.pop("pivot_rows", None)
    st.session_state.pop("pivot_cols", None)
    st.session_state.pop("pivot_filters", None)
    st.session_state.pop("pivot_vals", None)
    st.session_state.pop("pivot_adv", None)
    st.session_state.pop("pivot_agg", None)
    st.session_state.pop("pivot_fill", None)
    st.session_state.pop("merged_result", None)
    st.session_state.pop("merge_info", None)

    st.session_state["last_sheet_loc"] = sheet_loc
    st.session_state["last_sheet_ctrl"] = sheet_ctrl
    st.info("Sheet changed — please map columns again if needed.")

# =========================
# MAPPING UI
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
    "dept": ["Resource Dept", "Dept", "Department", "Resource Department"],
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

# Validate mapping – block Run until all required keys are valid
missing = validate_mapping(df_loc, df_ctrl, loc_map, ctrl_map)
run_disabled = bool(missing["Location"] or missing["Controlling"])

with st.expander("Diagnostics (preview)", expanded=False):
    if missing["Location"] or missing["Controlling"]:
        st.error("Missing/invalid mapping:")
        if missing["Location"]:
            st.write("Location:", missing["Location"])
        if missing["Controlling"]:
            st.write("Controlling:", missing["Controlling"])
    st.write("Location columns:", list(df_loc.columns))
    st.write("Controlling columns:", list(df_ctrl.columns))
    st.dataframe(df_loc.head(5), use_container_width=True)
    st.dataframe(df_ctrl.head(5), use_container_width=True)

# =========================
# NORMALIZATION (dynamic)
# =========================
st.markdown("---")
st.header("Normalization Rules")

use_norm = st.checkbox("Apply normalization (multi-column)", value=True)

st.session_state.setdefault("norm_loc_maps", {})
st.session_state.setdefault("norm_ctrl_maps", {})

tab_loc, tab_ctrl = st.tabs(["Location", "Controlling"])

with tab_loc:
    st.caption("Pick Location columns to normalize, then edit the from→to mapping table.")
    loc_cols = list(df_loc.columns)
    sel_loc_cols = st.multiselect(
        "Columns to normalize (Location)",
        options=loc_cols,
        default=[c for c in [loc_map.get("emp_type"), loc_map.get("emp_location")] if c in loc_cols]
    )

    for col in sel_loc_cols:
        st.markdown(f"**Column:** `{col}`")
        existing = st.session_state["norm_loc_maps"].get(col, {})
        edit_df = sample_mapping_table(df_loc, col, existing)
        edit_df = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_loc_editor_{keysafe(col)}")
        clean_map = {str(a): str(b) for a, b in edit_df.itertuples(index=False) if str(a).strip() != ""}
        st.session_state["norm_loc_maps"][col] = clean_map
        st.divider()

with tab_ctrl:
    st.caption("Pick Controlling columns to normalize (optional).")
    ctrl_cols = list(df_ctrl.columns)
    defaults_ctrl = [ctrl_map.get("emp_type_like"), ctrl_map.get("dept"), ctrl_map.get("gb"), ctrl_map.get("month")]
    sel_ctrl_cols = st.multiselect(
        "Columns to normalize (Controlling)",
        options=ctrl_cols,
        default=[c for c in defaults_ctrl if c in ctrl_cols]
    )

    for col in sel_ctrl_cols:
        st.markdown(f"**Column:** `{col}`")
        existing = st.session_state["norm_ctrl_maps"].get(col, {})
        edit_df = sample_mapping_table(df_ctrl, col, existing)
        edit_df = st.data_editor(edit_df, num_rows="dynamic", key=f"norm_ctrl_editor_{keysafe(col)}")
        clean_map = {str(a): str(b) for a, b in edit_df.itertuples(index=False) if str(a).strip() != ""}
        st.session_state["norm_ctrl_maps"][col] = clean_map
        st.divider()

# =========================
# CONFIRMATION (simple)
# =========================
st.markdown("---")
st.header("Confirm: Emp Type mapping")
st.caption("By default, Controlling → `emp_type_like` is treated as equivalent to Location → `emp_type`. Change the mapping above if needed.")
st.checkbox("Keep this assumption", value=True, key="confirm_emp_type_link")

# =========================
# LAYER TOGGLES + RUN
# =========================
st.markdown("---")
st.header("Mapping Layers")
colL = st.columns(4)
with colL[0]:
    L1 = st.checkbox("L1: GB + Dept + EmpType + Month", value=True)
with colL[1]:
    L2 = st.checkbox("L2: Dept + EmpType + Month", value=True)
with colL[2]:
    L3 = st.checkbox("L3: GB + Dept + Month", value=True)
with colL[3]:
    L4 = st.checkbox("L4: Dept + Month", value=True)
enabled_layers = {"L1": L1, "L2": L2, "L3": L3, "L4": L4}

st.markdown("---")
st.header("Run")
run = st.button("Run processing", type="primary", disabled=run_disabled,
                help="Select all required columns before running." if run_disabled else None)
reclear = st.button("Clear last result")

if reclear:
    st.session_state.pop("merged_result", None)
    st.session_state.pop("merge_info", None)
    st.success("Cleared previous result.")

if run:
    loc = df_loc.copy()
    ctrl = df_ctrl.copy()
    if use_norm:
        loc = normalize_columns(loc, st.session_state.get("norm_loc_maps", {}))
        ctrl = normalize_columns(ctrl, st.session_state.get("norm_ctrl_maps", {}))

    merged, info = run_fallback_merges_expand(ctrl, loc, loc_map, ctrl_map, enabled_layers)
    merged = compute_outputs(merged, ctrl_map, on_div0=on_div0)

    st.session_state["merged_result"] = merged
    st.session_state["merge_info"] = info

    st.success("Processed")

# =========================
# RESULTS SNAPSHOT
# =========================
if "merged_result" in st.session_state:
    st.subheader("Results snapshot")
    st.caption(f"Rows total: {len(st.session_state['merged_result'])} — Mapped per layer: {st.session_state.get('merge_info', {})}")
    st.dataframe(st.session_state["merged_result"].head(100), use_container_width=True)

# =========================
# PIVOT (dynamic)
# =========================
st.markdown("---")
st.header("Pivot Table")

if "merged_result" not in st.session_state:
    st.info("No data to pivot. Click **Run processing** first.")
else:
    merged = st.session_state["merged_result"]
    cols_all = list(merged.columns)

    # Excel-like defaults (only if the columns exist)
    default_rows = [c for c in [ctrl_map.get("gb")] if c in cols_all]
    default_cols = [c for c in [ctrl_map.get("month")] if c in cols_all]
    default_filters = [c for c in [ctrl_map.get("emp_type_like"), ctrl_map.get("dept")] if c in cols_all]
    default_values = [v for v in ["Capacity_Location"] if v in cols_all]

    # Persist pivot state
    st.session_state.setdefault("pivot_adv", False)
    st.session_state.setdefault("pivot_rows", default_rows)
    st.session_state.setdefault("pivot_cols", default_cols)
    st.session_state.setdefault("pivot_filters", default_filters)
    st.session_state.setdefault("pivot_vals", default_values)
    st.session_state.setdefault("pivot_agg", "sum")
    st.session_state.setdefault("pivot_fill", 0.0)

    with st.expander("Pivot configuration", expanded=True):
        c_top = st.columns([1,1,1,1])
        with c_top[0]:
            st.checkbox("Advanced layout", key="pivot_adv",
                        help="Enable to change Rows/Columns; disable to keep GB / Month fixed.")
        with c_top[1]:
            st.selectbox("Aggregation", ["sum", "mean", "count"],
                         index=["sum","mean","count"].index(st.session_state["pivot_agg"]),
                         key="pivot_agg")
        with c_top[2]:
            st.number_input("Fill blank with", value=float(st.session_state["pivot_fill"]),
                            step=1.0, key="pivot_fill")
        with c_top[3]:
            st.multiselect("Values", options=cols_all,
                           default=st.session_state["pivot_vals"],
                           key="pivot_vals")

        c_mid = st.columns(2)
        with c_mid[0]:
            st.multiselect("Rows", options=cols_all,
                           default=st.session_state["pivot_rows"],
                           key="pivot_rows",
                           disabled=not st.session_state["pivot_adv"])
        with c_mid[1]:
            st.multiselect("Columns", options=cols_all,
                           default=st.session_state["pivot_cols"],
                           key="pivot_cols",
                           disabled=not st.session_state["pivot_adv"])

        st.markdown("**Filters**")
        st.multiselect("Filter fields", options=cols_all,
                       default=st.session_state["pivot_filters"],
                       key="pivot_filters")

        # Per-filter value pickers (with "(All)")
        filter_value_keys = {}
        for fc in st.session_state["pivot_filters"]:
            uniq_vals = sorted(map(str, merged[fc].dropna().unique().tolist()))
            opt = ["(All)"] + uniq_vals
            key_name = f"flt_vals_{keysafe(fc)}"
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

    # Month ordering (guarded)
    month_col = ctrl_map.get("month")
    if month_col and (month_col in dfp.columns) and (
        month_col in st.session_state.get("pivot_rows", []) or month_col in st.session_state.get("pivot_cols", [])
    ):
        dfp = ensure_month_order(dfp, month_col)

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
        )
        pivot_df = pivot_df.reset_index()
        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [" | ".join([str(x) for x in tup if str(x) != ""])
                                for tup in pivot_df.columns.values]

        st.subheader("Pivot result")
        st.dataframe(pivot_df, use_container_width=True)

        # Download pivot
        pivot_buf = io.BytesIO()
        with pd.ExcelWriter(pivot_buf, engine="xlsxwriter") as writer:
            notes = pd.DataFrame({
                "Filter Field": st.session_state["pivot_filters"],
                "Selected": [
                    ", ".join([v for v in st.session_state[filter_value_keys[f]]])
                    for f in st.session_state["pivot_filters"]
                ],
            })
            pivot_df.to_excel(writer, sheet_name="Pivot", index=False)
            notes.to_excel(writer, sheet_name="Notes", index=False)
        st.download_button(
            "Download Pivot (Excel)",
            data=pivot_buf.getvalue(),
            file_name="pivot_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Pivot error: {e}")

# =========================
# DOWNLOAD RESULT
# =========================
if "merged_result" in st.session_state:
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        st.session_state["merged_result"].drop(columns=["__mapped__"], errors="ignore")\
            .to_excel(writer, sheet_name="Result", index=False)
    st.download_button(
        label="Download Excel result",
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
