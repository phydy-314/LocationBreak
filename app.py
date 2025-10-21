# app.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG & DEFAULTS
# =========================
st.set_page_config(page_title="Stimulation không còn đao khổ nữa:))", layout="wide")

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
@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name=None):
    xls = pd.ExcelFile(file)
    if sheet_name is None:
        return {sn: xls.parse(sn) for sn in xls.sheet_names}
    else:
        return {sheet_name: xls.parse(sheet_name)}

def normalize_values(df_loc, col_emp_type, col_emp_loc, norm_cfg):
    df = df_loc.copy()
    if col_emp_type in df.columns:
        df[col_emp_type] = (
            df[col_emp_type].astype(str)
            .map(norm_cfg.get("emp_type_map", {}))
            .fillna(df[col_emp_type].astype(str))
        )
    if col_emp_loc in df.columns:
        df[col_emp_loc] = (
            df[col_emp_loc].astype(str)
            .map(norm_cfg.get("emp_location_map", {}))
            .fillna(df[col_emp_loc].astype(str))
        )
    return df

def compute_ratio_expanding(df_loc, keys, emp_loc_col, cap_col):
    """
    Ratio theo KEYS + Emp Location. Nếu tổng theo keys = 0 -> Ratio = 0 (coi là match để nổ hàng).
    Trả về: keys + [emp_loc_col, 'Ratio']
    """
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
    """
    Gom các biến thể cột (base + *_x/_y/...) về 1 cột chuẩn `canonical`.
    Ưu tiên theo thứ tự trong base_names (trước = ưu tiên cao hơn).
    """
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
    Tách → map → append theo từng lớp, nổ theo Emp Location.
    Sau mỗi lớp chỉ append phần đã map; phần còn NaN đi lớp sau.
    Sau L4: không append phần còn blank.
    Có gom cột (_x/_y) về cột chuẩn sau mỗi merge.
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

        # Tính capacity_location & tách matched/unmatched
        work_merged["Capacity_Location_tmp"] = work_merged[m_ctrl["capacity"]] * work_merged["Ratio"]
        matched_mask = work_merged["Ratio"].notna()

        matched   = work_merged.loc[matched_mask].copy()
        unmatched = work_merged.loc[~matched_mask].copy()

        matched["Capacity_Location"] = matched["Capacity_Location_tmp"]
        matched["MappedLayer"] = layer_tag

        # Căn cột chuẩn & append
        if emp_loc_col not in base_df.columns:
            base_df[emp_loc_col] = np.nan
        matched_aligned = matched.reindex(columns=base_df.columns, fill_value=np.nan)
        base_df = pd.concat([base_df, matched_aligned], ignore_index=True)
        return base_df, unmatched, int(matched_mask.sum())

    # keys cho Location theo lớp
    keys_L1_loc = [m_loc["gb"], m_loc["dept"], m_loc["emp_type"], m_loc["month"]]
    keys_L2_loc = [m_loc["dept"], m_loc["emp_type"], m_loc["month"]]
    keys_L3_loc = [m_loc["gb"], m_loc["dept"], m_loc["month"]]
    keys_L4_loc = [m_loc["dept"], m_loc["month"]]

    # L1
    work_next = pd.DataFrame()
    if enabled_layers.get("L1", True):
        base, work_next, cnt = _layer(base, keys_L1_loc, "L1"); merge_info["L1"] = cnt
    else:
        work_next = base.loc[base["Capacity_Location"].isna()].copy()
    # L2
    if enabled_layers.get("L2", True) and not work_next.empty:
        base = pd.concat([base, work_next], ignore_index=True)
        base, work_next, cnt = _layer(base, keys_L2_loc, "L2"); merge_info["L2"] = cnt
    # L3
    if enabled_layers.get("L3", True) and not work_next.empty:
        base = pd.concat([base, work_next], ignore_index=True)
        base, work_next, cnt = _layer(base, keys_L3_loc, "L3"); merge_info["L3"] = cnt
    # L4
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

# =========================
# UI
# =========================
st.title("Stimulation không còn đao khổ nữa:))")

with st.sidebar:
    st.header("Upload Excel")
    excel_file = st.file_uploader("Upload .xlsx", type=["xlsx", "xlsm"])
    on_div0 = st.selectbox("When divide-by-zero:", ["zero", "blank"], index=0)

if not excel_file:
    st.info("Upload your Excel to get started.")
    st.stop()

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

st.markdown("---")
st.header("Map Columns")

loc_map = {}
ctrl_map = {}

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
        g = guess(df_loc.columns, loc_candidates.get(key, [])) or st.selectbox(
            f"Select column for `{key}`", options=[None] + list(df_loc.columns), index=0, key=f"loc_{key}_first"
        )
        loc_map[key] = st.selectbox(
            f"{key}",
            options=[None] + list(df_loc.columns),
            index=(list([None] + list(df_loc.columns)).index(g) if g in ([None] + list(df_loc.columns)) else 0),
            key=f"loc_{key}",
        )
with c2:
    st.subheader("Controlling columns")
    for key in REQUIRED_KEYS_CTRL:
        g = guess(df_ctrl.columns, ctrl_candidates.get(key, [])) or st.selectbox(
            f"Select column for `{key}`", options=[None] + list(df_ctrl.columns), index=0, key=f"ctrl_{key}_first"
        )
        ctrl_map[key] = st.selectbox(
            f"{key}",
            options=[None] + list(df_ctrl.columns),
            index=(list([None] + list(df_ctrl.columns)).index(g) if g in ([None] + list(df_ctrl.columns)) else 0),
            key=f"ctrl_{key}",
        )

missing_loc = [k for k in REQUIRED_KEYS_LOCATION if not loc_map.get(k)]
missing_ctrl = [k for k in REQUIRED_KEYS_CTRL if not ctrl_map.get(k)]
if missing_loc or missing_ctrl:
    st.error(f"Missing mapping: Location -> {missing_loc} | Controlling -> {missing_ctrl}")
    st.stop()

st.markdown("---")
st.header("Normalization Rules (optional)")
use_norm = st.checkbox("Apply normalization for Emp Type & Emp Location", value=True)
if "norm_cfg" not in st.session_state:
    st.session_state["norm_cfg"] = DEFAULT_NORMALIZATION.copy()

with st.expander("Edit normalization maps"):
    st.caption("Left column = from, Right column = to")
    emp_map_df = pd.DataFrame(
        list(st.session_state["norm_cfg"]["emp_type_map"].items()), columns=["from", "to"]
    )
    emp_map_df = st.data_editor(emp_map_df, num_rows="dynamic", key="emp_type_editor")
    st.session_state["norm_cfg"]["emp_type_map"] = dict(emp_map_df.values)

    loc_map_df = pd.DataFrame(
        list(st.session_state["norm_cfg"]["emp_location_map"].items()), columns=["from", "to"]
    )
    loc_map_df = st.data_editor(loc_map_df, num_rows="dynamic", key="emp_loc_editor")
    st.session_state["norm_cfg"]["emp_location_map"] = dict(loc_map_df.values)

# Popup xác nhận: Header Service ≡ Emp Type (giữ behavior cũ)
if "confirm_emp_type_link" not in st.session_state:
    st.session_state["confirm_emp_type_link"] = False

@st.dialog("Confirm: Emp Type mapping")
def confirm_dialog():
    st.write(
        "By default, **Controlling → `emp_type_like`** (ví dụ `Header Service`) được xem tương đương **Location → `emp_type`**.\n\n"
        "Bạn muốn giữ giả định này? Có thể chỉnh lại cột ở bước 3 nếu cần."
    )
    if st.button("Giữ giả định"):
        st.session_state["confirm_emp_type_link"] = True
        st.rerun()
    if st.button("Mình sẽ tự đổi cột"):
        st.session_state["confirm_emp_type_link"] = True
        st.rerun()

if not st.session_state["confirm_emp_type_link"]:
    confirm_dialog()

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
st.header("6) Run")
run = st.button("Run processing")

if run:
    loc = df_loc.copy()
    ctrl = df_ctrl.copy()

    if use_norm:
        loc = normalize_values(loc, loc_map["emp_type"], loc_map["emp_location"], st.session_state["norm_cfg"])

    merged, info = run_fallback_merges_expand(
        ctrl,
        loc,
        loc_map,
        ctrl_map,
        enabled_layers,
    )

    # Tính output
    merged = compute_outputs(merged, ctrl_map, on_div0=on_div0)

    st.subheader("Results")
    st.write("Rows total:", len(merged))
    st.write("Mapped per layer:", info)
    st.dataframe(merged.head(100), use_container_width=True)

    # =========================
    # PIVOT TABLE (giống Excel)
    # =========================
    st.markdown("---")
    st.header("Pivot Table quick check")

    with st.expander("Cấu hình Pivot"):
        cols_all = list(merged.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            pivot_rows = st.multiselect("Rows", options=cols_all, default=[ctrl_map["gb"], ctrl_map["dept"]])
        with c2:
            pivot_cols = st.multiselect("Columns", options=cols_all, default=[ctrl_map["month"]])
        with c3:
            pivot_vals = st.multiselect("Values", options=cols_all, default=["Capacity_Location", "Budget Location"])

        agg_choice = st.selectbox("Aggregation", ["sum", "mean", "count"], index=0)
        fillna_val = st.number_input("Fill blank with", value=0.0, step=1.0)

    if pivot_rows or pivot_cols or pivot_vals:
        try:
            aggfunc = {"sum": np.sum, "mean": np.mean, "count": "count"}[agg_choice]
            pivot_df = pd.pivot_table(
                merged,
                index=pivot_rows if pivot_rows else None,
                columns=pivot_cols if pivot_cols else None,
                values=pivot_vals if pivot_vals else None,
                aggfunc=aggfunc,
                fill_value=fillna_val,
                dropna=False,
            )
            # Đưa MultiIndex về cột phẳng cho dễ xem/tải
            if isinstance(pivot_df.index, pd.MultiIndex):
                pivot_df = pivot_df.reset_index()
            else:
                pivot_df = pivot_df.reset_index()

            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [" | ".join([str(x) for x in tup if str(x) != ""])
                                    for tup in pivot_df.columns.values]

            st.subheader("Pivot result")
            st.dataframe(pivot_df, use_container_width=True)

            # Tải pivot
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

    # =========================
    # DOWNLOAD KẾT QUẢ
    # =========================
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        merged.drop(columns=["__mapped__"], errors="ignore").to_excel(writer, sheet_name="Result", index=False)
    st.download_button(
        label="Download Excel result",
        data=out_buf.getvalue(),
        file_name="mapping_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.success("Done!")
# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; font-size:13px; color:#666; margin-top:8px;">
        Crafted with care by the <strong>BGSV/CTG Data Team</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)
