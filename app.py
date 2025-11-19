# -*- coding: utf-8 -*-
"""
Ace Pigeon (Club) – Web App (Streamlit)
Release (2025-11-11) – with Best Loft Ranking + simple PDF builder

Rules:
- Ignore rankno from get_race
- Sort by SPEED (desc), RANK=1..n
- N = round(total_marked * percentage), at least 1
- coefficient = first_place_points / N
- rank r points = first_place_points - (r-1)*coefficient; r > N → 0; min 0

Features:
- get_racelist / get_race / get_marked (case-insensitive keys)
- Single race & Ace aggregation export CSV / PDF (ReportLab simple builder)
- Best Loft Ranking: sum of top N birds per loft (from Ace Pigeon result)
- session_state to keep list & selections
- Language switch (default English)
"""

import io
import os
import sys

def _resource_path(name: str) -> str:
    """Return absolute path to resource, works for dev and PyInstaller onefile."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, name)

import json
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st

# ReportLab base imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet

# ============ PDF（reportlab） flag like OLR_app ============
try:
    from reportlab.lib import colors as _rl_colors
    from reportlab.lib.pagesizes import A4 as _A4, landscape as _landscape
    from reportlab.lib.styles import getSampleStyleSheet as _getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate as _SimpleDocTemplate,
        Table as _Table,
        TableStyle as _TableStyle,
        Paragraph as _Paragraph,
        Spacer as _Spacer,
    )
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---- ReportLab MD5 FIPS/usedforsecurity compatibility shim ----
import hashlib
try:
    from reportlab.pdfbase import pdfdoc as _pdfdoc

    def _md5_compat(*args, **kwargs):
        data = b""
        if args:
            data = args[0]
        return hashlib.md5(data)

    _pdfdoc.md5 = _md5_compat  # override to ignore unsupported kwargs
except Exception:
    pass
# ----------------------------------------------------------------


def build_pdf_from_df(df: pd.DataFrame, title: str) -> bytes:
    """Simplified ReportLab PDF builder like OLR_app (no external fonts)."""
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")
    from io import BytesIO

    buffer = BytesIO()
    doc = _SimpleDocTemplate(
        buffer,
        pagesize=_landscape(_A4),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    styles = _getSampleStyleSheet()
    elems = [_Paragraph(title, styles["Title"]), _Spacer(1, 12)]
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = _Table(data, repeatRows=1)
    table.setStyle(
        _TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _rl_colors.black),
                ("TEXTCOLOR", (0, 0), (-1, 0), _rl_colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, _rl_colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_rl_colors.whitesmoke, _rl_colors.lightgrey]),
            ]
        )
    )
    elems.append(table)
    doc.build(elems)
    buffer.seek(0)
    return buffer.read()


# 這行其實現在沒用到（df_to_pdf_bytes 不再被呼叫），只是舊碼保留不影響執行
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

# ----- ReportLab md5 compatibility (older OpenSSL) -----
try:
    from reportlab.pdfbase import pdfdoc as rl_pdfdoc
    import hashlib as _hashlib

    try:
        rl_pdfdoc.md5(usedforsecurity=False)  # type: ignore
    except TypeError:
        rl_pdfdoc.md5 = _hashlib.md5
except Exception:
    pass
# ------------------------------------------------------

# ---------- UI ----------
st.set_page_config(page_title="Ace Pigeon (Club) List", layout="wide")
st.title("🏆 Ace Pigeon (Club) List")

# Language (default English)
lang = st.sidebar.selectbox("Language / 語言", ["English", "中文"], index=0)

TEXTS = {
    "中文": {
        "help_title": "使用說明 / Scoring 說明",
        "help_body": (
            "- 輸入你的Club No., 年份（跨年可填兩個）\n"
            "- 抓取賽事清單\n"
            "- 勾選賽事計算個別分數（可多選）\n"
            "- 總匯所有分數\n"
            "- 可匯出 CSV / PDF。"
        ),
        "sidebar_header": "API 與查詢參數",
        "first_points": "第一名的分數 (1st place)",
        "percent": "取多少名次的百分比 (%)",
        "fetch_list": "📥 抓取賽事清單",
        "need_params": "請輸入 clubno / 年份(至少一個)。",
        "no_races": "未取得任何賽事，請確認參數或 API 可用性。",
        "step2": "② 選取賽事 + 單場成績計算",
        "select_label": "選擇賽事 (可多選)",
        "calc_score": "計算分數",
        "auto_marked": "總參賽鴿數 (自動)",
        "manual_marked": "無法自動取得總參賽鴿數，請手動輸入",
        "calc_done": "計算完成",
        "download_csv": "⬇️ 下載 CSV",
        "download_pdf": "⬇️ 下載 PDF",
        "step3": "③ Ace Pigeon List (多賽事加總)",
        "agg_btn": "計算彙總分數 (Ace Pigeon List)",
        "agg_done": "彙總完成！",
        "agg_csv": "⬇️ 下載 CSV (Ace Pigeon List)",
        "agg_pdf": "⬇️ 下載 PDF (Ace Pigeon List)",
        "pdf_fail": "PDF 產生失敗（ReportLab 無法使用）。請先下載 CSV。",
        "ver": "版本：2025-11-11（忽略 rankno + 依速度排序 + 等差遞減計分；簡易 PDF）",
        # Best Loft Ranking 相關文字
        "best_loft_sidebar": "Best Loft Ranking – 每舍取前N羽",
        "best_loft_n": "每舍取幾羽計算最佳舍排名",
        "best_loft_title": "Best Loft Ranking (依 Ace Pigeon 積分)",
        "best_loft_csv": "⬇️ 下載 CSV (Best Loft Ranking)",
        "best_loft_pdf": "⬇️ 下載 PDF (Best Loft Ranking)",
    },
    "English": {
        "help_title": "How it works",
        "help_body": (
            "- Enter your Club No. and year (two years if spanning)\n"
            "- Fetch the race list\n"
            "- Select races (multi-select) and compute single-race scores\n"
            "- Aggregate to Ace Pigeon List\n"
            "- Export to CSV / PDF."
        ),
        "sidebar_header": "API & Parameters",
        "first_points": "1st place points",
        "percent": "Percentage for scoring (%)",
        "fetch_list": "📥 Fetch Race List",
        "need_params": "Please provide clubno and at least one year.",
        "no_races": "No races found. Please check parameters or API availability.",
        "step2": "② Select races & compute single-race scores",
        "select_label": "Select races (multi-select)",
        "calc_score": "Compute Scores",
        "auto_marked": "Total marked (auto)",
        "manual_marked": "Total marked not available, please input manually",
        "calc_done": "Done",
        "download_csv": "⬇️ Download CSV",
        "download_pdf": "⬇️ Download PDF",
        "step3": "③ Ace Pigeon List (aggregate)",
        "agg_btn": "Compute Aggregated (Ace Pigeon List)",
        "agg_done": "Aggregated!",
        "agg_csv": "⬇️ Download CSV (Ace Pigeon List)",
        "agg_pdf": "⬇️ Download PDF (Ace Pigeon List)",
        "pdf_fail": "PDF generation failed (ReportLab unavailable). Please download CSV instead.",
        "ver": "Version: 2025-11-11 (ignore rankno + speed sort + arithmetic scoring; simple PDF)",
        # Best Loft Ranking texts
        "best_loft_sidebar": "Best Loft Ranking – top N birds per loft",
        "best_loft_n": "Top N birds per loft",
        "best_loft_title": "Best Loft Ranking (from Ace points)",
        "best_loft_csv": "⬇️ Download CSV (Best Loft Ranking)",
        "best_loft_pdf": "⬇️ Download PDF (Best Loft Ranking)",
    },
}

def T(key: str) -> str:
    return TEXTS[lang].get(key, key)


with st.expander(T("help_title"), expanded=False):
    st.markdown(TEXTS[lang]["help_body"])

# ---------- Sidebar (fixed credentials embedded) ----------
st.sidebar.header(TEXTS[lang]["sidebar_header"])
uname = "RafaelES"
ukey = "ae866e78944cabad"
clubno = st.sidebar.text_input("clubno", value="")
col_y1, col_y2 = st.sidebar.columns(2)
year1 = col_y1.text_input("year1", value="2025")
year2 = col_y2.text_input("year2", value="")

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring")
first_place_points = st.sidebar.number_input(
    TEXTS[lang]["first_points"], min_value=1.0, value=100.0, step=1.0
)
percent_with_points = st.sidebar.number_input(
    TEXTS[lang]["percent"],
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
)

# ---- Best Loft Ranking 參數 ----
st.sidebar.subheader(TEXTS[lang].get("best_loft_sidebar", "Best Loft Ranking"))
best_loft_n = st.sidebar.number_input(
    TEXTS[lang]["best_loft_n"], min_value=1, value=3, step=1
)

# ---------- API ----------
BASE = "https://www.topigeon.com/api/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

@st.cache_data(show_spinner=False)
def call_api(params: Dict[str, str]) -> Optional[dict]:
    try:
        r = requests.get(BASE, params=params, timeout=30, headers={"User-Agent": UA})
        text = r.text.strip()
        try:
            return r.json()
        except Exception:
            if text.startswith("{"):
                try:
                    return json.loads(text)
                except Exception:
                    pass
            return {"raw": text, "_status": r.status_code, "_len": len(text)}
    except Exception as e:
        return {"error": str(e)}

def _extract_list_like(payload: object) -> Optional[list]:
    if isinstance(payload, dict):
        for key in ("data", "Data", "DATA", "list", "items"):
            v = payload.get(key)  # type: ignore
            if isinstance(v, list):
                return v
    elif isinstance(payload, list):
        return payload
    return None

@st.cache_data(show_spinner=False)
def get_racelist(clubno: str, years: List[str], uname: str, ukey: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for y in (s.strip() for s in years if s):
        items = None
        variants = [
            dict(act="get_racelist", clubno=clubno, raceyear=y, uname=uname, ukey=ukey),
            dict(act="get_racelist", clubno=clubno, raceyear=y, uname=uname, ukey=ukey, APP="Y"),
        ]
        for params in variants:
            data = call_api(params)
            if not data:
                continue
            items = _extract_list_like(data)
            if items:
                break
        if not items:
            continue
        df = pd.DataFrame(items)
        df["raceyear"] = y
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    col_map = {
        "raceno": "raceno",
        "race_no": "raceno",
        "raceid": "raceno",
        "raceyear": "raceyear",
        "race_year": "raceyear",
        "racename": "racename",
        "race_name": "racename",
        "title": "racename",
        "date": "date",
        "racetime": "date",
        "racedate": "date",
    }
    out = out.rename(columns={k: v for k, v in col_map.items() if k in out.columns})
    keep = [c for c in ["raceno", "raceyear", "racename", "date"] if c in out.columns]
    other = [c for c in out.columns if c not in keep]
    return out[keep + other]

@st.cache_data(show_spinner=False)
def get_race(
    raceno: str, raceyear: str, clubno: str, uname: str, ukey: str
) -> pd.DataFrame:
    params = dict(
        act="get_race",
        raceno=raceno,
        raceyear=raceyear,
        clubno=clubno,
        uname=uname,
        ukey=ukey,
        APP="Y",
    )
    data = call_api(params)
    if not data:
        return pd.DataFrame()
    items = _extract_list_like(data)
    if items is None and isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame(items or [])

@st.cache_data(show_spinner=False)
def get_marked_count(
    raceno: str, markedyear: str, clubno: str, uname: str, ukey: str
) -> Optional[int]:
    params = dict(
        act="get_marked",
        uname=uname,
        ukey=ukey,
        raceno=raceno,
        markedyear=markedyear,
        clubno=clubno,
    )
    data = call_api(params)
    items = _extract_list_like(data) if data else None
    return len(items) if isinstance(items, list) else None

# ---------- Scoring ----------
def _to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def standardize_race_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map to RANK, LOFT NAME, PRING_NO, SPEED; ignore API rankno; sort by SPEED."""
    if df.empty:
        return df
    used_map = {}
    for src, tgt in [
        ("rank", "RANK"),
        ("rankno", "RANK"),
        ("順位", "RANK"),
        ("pos", "RANK"),
        ("RANK", "RANK"),
        ("loftname", "LOFT NAME"),
        ("loft_name", "LOFT NAME"),
        ("loft", "LOFT NAME"),
        ("LOFT_NAME", "LOFT NAME"),
        ("鴿舍", "LOFT NAME"),
        ("pring_no", "PRING_NO"),
        ("pringno", "PRING_NO"),
        ("環號", "PRING_NO"),
        ("PRING_NO", "PRING_NO"),
        ("flyspeed", "SPEED"),
        ("speed", "SPEED"),
        ("速度", "SPEED"),
        ("v", "SPEED"),
        ("SPEED", "SPEED"),
    ]:
        if src in df.columns:
            used_map[src] = tgt
    sdf = df.rename(columns=used_map).copy()
    for col in ["RANK", "LOFT NAME", "PRING_NO", "SPEED"]:
        if col not in sdf.columns:
            sdf[col] = None
    sdf["SPEED"] = sdf["SPEED"].map(_to_float)
    sdf = sdf.sort_values(by=["SPEED"], ascending=False, na_position="last").reset_index(
        drop=True
    )
    sdf["RANK"] = range(1, len(sdf) + 1)
    return sdf[["RANK", "LOFT NAME", "PRING_NO", "SPEED"]]

def compute_points_table(
    sdf: pd.DataFrame, total_marked: int, first_points: float, pct_rank: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    N = max(int(round((total_marked or 0) * (pct_rank / 100.0))), 1)
    first_points = float(first_points)
    coefficient = first_points / float(N)

    def score_by_rank(r: int) -> float:
        if r is None or r <= 0 or r > N:
            return 0.0
        return max(first_points - (r - 1) * coefficient, 0.0)

    out = sdf.copy()
    out["POINT"] = out["RANK"].map(score_by_rank)
    meta = dict(
        total_marked=int(total_marked or 0),
        N=N,
        coefficient=coefficient,
        first_points=first_points,
        pct_rank=float(pct_rank),
    )
    return out, meta

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# ---------- UI: fetch & compute ----------
st.subheader(
    "① " + ("抓取賽事清單 (get_racelist)" if lang == "中文" else "Fetch race list (get_racelist)")
)

if "racelist" not in st.session_state:
    st.session_state["racelist"] = pd.DataFrame()
if "selected_rows" not in st.session_state:
    st.session_state["selected_rows"] = []

def _set_selected_rows():
    st.session_state["selected_rows"] = st.session_state.get("race_select", [])

fetch_clicked = st.button(TEXTS[lang]["fetch_list"], use_container_width=True)
race_df = st.session_state["racelist"]

if fetch_clicked:
    years = [y for y in [year1, year2] if y]
    if not (clubno and years):
        st.error(TEXTS[lang]["need_params"])
    else:
        with st.spinner("Loading..."):
            df_new = get_racelist(clubno, years, uname, ukey)
        if df_new.empty:
            st.warning(TEXTS[lang]["no_races"])
        race_df = df_new
        st.session_state["racelist"] = df_new

if not race_df.empty:
    st.dataframe(race_df, use_container_width=True)
    st.subheader(TEXTS[lang]["step2"])
    key_cols = [c for c in ["raceno", "raceyear", "racename"] if c in race_df.columns] or [
        race_df.columns[0]
    ]
    display_df = race_df[key_cols].copy()
    display_df["_display"] = display_df.apply(
        lambda r: " | ".join([str(r[c]) for c in key_cols]), axis=1
    )

    selected_rows = st.multiselect(
        TEXTS[lang]["select_label"],
        options=list(display_df.index),
        format_func=lambda i: display_df.loc[i, "_display"],
        default=st.session_state.get("selected_rows", []),
        key="race_select",
        on_change=_set_selected_rows,
    )

    if st.session_state.get("selected_rows"):
        st.markdown(
            "**Single race / Export**" if lang == "English" else "**單場計算 / 匯出**"
        )
        for idx in st.session_state["selected_rows"]:
            row = race_df.loc[idx]
            raceno_val = str(row.get("raceno", ""))
            raceyear_val = str(row.get("raceyear", ""))
            racename_val = str(row.get("racename", raceno_val))
            header = f"📊 {racename_val} ({raceyear_val}) / raceno={raceno_val}"
            with st.expander(header, expanded=False):
                colA, colB, colC = st.columns([1, 1, 2])
                with colA:
                    run_btn = st.button(TEXTS[lang]["calc_score"], key=f"calc_{idx}")
                with colB:
                    st.caption(
                        f"{TEXTS[lang]['first_points']}: {first_place_points}｜{TEXTS[lang]['percent']}: {percent_with_points}%"
                    )
                total_marked_auto = get_marked_count(
                    raceno_val, raceyear_val, clubno, uname, ukey
                )
                with colC:
                    if total_marked_auto is None:
                        total_marked_input = st.number_input(
                            f"{TEXTS[lang]['manual_marked']} (raceno={raceno_val})",
                            min_value=1,
                            value=100,
                            step=1,
                            key=f"marked_{idx}",
                        )
                        total_marked = int(total_marked_input)
                    else:
                        total_marked = int(total_marked_auto)
                        st.info(f"{TEXTS[lang]['auto_marked']}: {total_marked}")

                if run_btn:
                    with st.spinner("Computing..."):
                        df_raw = get_race(
                            raceno_val, raceyear_val, clubno, uname, ukey
                        )
                        sdf = standardize_race_df(df_raw)
                        if sdf.empty:
                            st.warning(
                                "No valid results."
                                if lang == "English"
                                else "此賽事未取得有效成績資料。"
                            )
                        else:
                            out_df, meta = compute_points_table(
                                sdf,
                                total_marked,
                                first_place_points,
                                percent_with_points,
                            )
                            st.success(
                                f"{TEXTS[lang]['calc_done']}: total={meta['total_marked']}, N={meta['N']}, coef={meta['coefficient']:.4f}"
                            )
                            st.dataframe(out_df, use_container_width=True)
                            st.download_button(
                                label=TEXTS[lang]["download_csv"],
                                data=df_to_csv_bytes(out_df),
                                file_name=f"race_{raceno_val}_{raceyear_val}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key=f"csv_{idx}",
                            )
                            try:
                                pdf_bytes = build_pdf_from_df(
                                    out_df,
                                    title=f"{racename_val} ({raceyear_val}) – Score",
                                )
                                st.download_button(
                                    label=TEXTS[lang]["download_pdf"],
                                    data=pdf_bytes,
                                    file_name=f"race_{raceno_val}_{raceyear_val}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key=f"pdf_{idx}",
                                )
                            except Exception:
                                st.info(TEXTS[lang]["pdf_fail"])

# ---------- Ace Pigeon List (aggregate) ----------
st.markdown("---")
st.subheader(TEXTS[lang]["step3"])

if st.session_state.get("selected_rows"):
    if st.button(TEXTS[lang]["agg_btn"], type="primary"):
        total_tables: List[pd.DataFrame] = []
        with st.spinner("Aggregating..."):
            for idx in st.session_state["selected_rows"]:
                row = race_df.loc[idx]
                raceno_val = str(row.get("raceno", ""))
                raceyear_val = str(row.get("raceyear", ""))
                racename_val = str(row.get("racename", raceno_val))
                df_raw = get_race(
                    raceno_val, raceyear_val, clubno, uname, ukey
                )
                sdf = standardize_race_df(df_raw)
                if sdf.empty:
                    continue
                total_marked_auto = get_marked_count(
                    raceno_val, raceyear_val, clubno, uname, ukey
                )
                total_marked = (
                    int(total_marked_auto) if total_marked_auto is not None else 100
                )
                out_df, _ = compute_points_table(
                    sdf, total_marked, first_place_points, percent_with_points
                )
                out_df["_race_tag"] = f"{racename_val}({raceyear_val})"
                total_tables.append(out_df)

        if not total_tables:
            st.warning(
                "Nothing to aggregate."
                if lang == "English"
                else "無可彙總的成績資料。"
            )
        else:
            merged = pd.concat(total_tables, ignore_index=True)
            grp = merged.groupby(["LOFT NAME", "PRING_NO"], dropna=False)

            agg = pd.DataFrame(
                {
                    "POINT": grp["POINT"].sum(),
                    "RACES": grp.size(),
                    "AVG_SPEED": grp["SPEED"].mean(),
                }
            ).reset_index()

            agg["AVG_SPEED"] = agg["AVG_SPEED"].round(3)
            agg["POINT"] = agg["POINT"].round(4)

            agg = agg.sort_values(
                ["POINT", "AVG_SPEED"], ascending=[False, False]
            ).reset_index(drop=True)
            agg.insert(0, "RANK", range(1, len(agg) + 1))
            agg = agg[
                ["RANK", "LOFT NAME", "PRING_NO", "POINT", "RACES", "AVG_SPEED"]
            ]

            st.success(TEXTS[lang]["agg_done"])
            st.dataframe(agg, use_container_width=True)
            st.download_button(
                label=TEXTS[lang]["agg_csv"],
                data=df_to_csv_bytes(agg),
                file_name="ace_pigeon_list.csv",
                mime="text/csv",
                use_container_width=True,
                key="ace_csv",
            )
            try:
                pdf_total = build_pdf_from_df(
                    agg, title="Ace Pigeon List – Total Points"
                )
                st.download_button(
                    label=TEXTS[lang]["agg_pdf"],
                    data=pdf_total,
                    file_name="ace_pigeon_list.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="ace_pdf",
                )
            except Exception:
                st.info(TEXTS[lang]["pdf_fail"])

            # ---------- Best Loft Ranking (from Ace Pigeon agg) ----------
            # 每一個 LOFT NAME 取該舍積分最高的 N 羽，合計後再做排名
            if best_loft_n and int(best_loft_n) > 0:
                N = int(best_loft_n)
                loft_groups = agg.groupby("LOFT NAME", dropna=False)
                rows = []
                for loft_name, g in loft_groups:
                    # 依 POINT、AVG_SPEED 做排序，取前 N 羽
                    g_sorted = g.sort_values(
                        ["POINT", "AVG_SPEED"], ascending=[False, False]
                    )
                    top_g = g_sorted.head(N)
                    total_point = top_g["POINT"].sum()
                    birds_used = len(top_g)
                    rows.append(
                        {
                            "LOFT NAME": loft_name,
                            "BEST_TOTAL_POINT": total_point,
                            "BIRDS_USED": birds_used,
                        }
                    )

                if rows:
                    best_df = pd.DataFrame(rows)
                    best_df["BEST_TOTAL_POINT"] = best_df["BEST_TOTAL_POINT"].round(4)
                    best_df = best_df.sort_values(
                        "BEST_TOTAL_POINT", ascending=False
                    ).reset_index(drop=True)
                    best_df.insert(0, "RANK", range(1, len(best_df) + 1))

                    st.subheader(TEXTS[lang]["best_loft_title"])
                    st.dataframe(best_df, use_container_width=True)

                    st.download_button(
                        label=TEXTS[lang]["best_loft_csv"],
                        data=df_to_csv_bytes(best_df),
                        file_name=f"best_loft_ranking_top{N}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="best_loft_csv",
                    )
                    try:
                        pdf_best = build_pdf_from_df(
                            best_df, title=f"Best Loft Ranking – Top {N} Birds"
                        )
                        st.download_button(
                            label=TEXTS[lang]["best_loft_pdf"],
                            data=pdf_best,
                            file_name=f"best_loft_ranking_top{N}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="best_loft_pdf",
                        )
                    except Exception:
                        st.info(TEXTS[lang]["pdf_fail"])

st.caption(TEXTS[lang]["ver"])
