import streamlit as st

import pandas as pd
import numpy as np
import os
import re
import math
import string
import datetime
import json
import time
import requests
from base64 import b64encode
from groq import Groq
from io import BytesIO

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Enterprise AI Platform", layout="wide")

# -----------------------------------
# FULL CSS
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@600;700&family=Orbitron:wght@700;900&family=Space+Mono:wght@400;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.built-by-banner { display: flex; align-items: center; justify-content: flex-end; gap: 8px; padding: 6px 16px 0 0; margin-bottom: -6px; }
.built-by-banner .byline { font-size: 11px; color: #999; letter-spacing: 0.8px; text-transform: uppercase; }
.built-by-banner .author { font-family: 'Rajdhani', sans-serif; font-size: 15px; font-weight: 700; color: #B31B1B; letter-spacing: 1px; }
.built-by-banner .dot { width: 6px; height: 6px; background: #FFC72C; border-radius: 50%; display: inline-block; }

.main-header { background: linear-gradient(135deg, #B31B1B 0%, #7a1212 100%); padding: 22px 28px; border-radius: 10px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 15px rgba(179,27,27,0.3); }
.main-header h1 { color: #FFC72C; margin: 0; font-family: 'Rajdhani', sans-serif; font-size: 28px; font-weight: 700; letter-spacing: 1px; }
.main-header .header-sub { color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px; letter-spacing: 0.5px; }
.main-header .version-badge { background: rgba(255,199,44,0.15); border: 1px solid rgba(255,199,44,0.4); color: #FFC72C; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 1px; }

.section-title { color: #B31B1B; font-weight: 600; font-size: 16px; margin-top: 20px; text-transform: uppercase; letter-spacing: 0.5px; }

.stButton>button { background-color: #B31B1B; color: white; font-weight: bold; border-radius: 6px; }
.stButton>button:hover { background-color: #8E1414; color: #FFC72C; }

.metric-row { display: flex; gap: 12px; margin: 16px 0 8px 0; flex-wrap: wrap; }
.metric-box { background: white; border: 1px solid #E8E8E8; border-top: 3px solid #B31B1B; border-radius: 8px; padding: 14px 18px; min-width: 120px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); flex: 1; }
.metric-box .metric-value { font-size: 28px; font-weight: 700; color: #B31B1B; font-family: 'Rajdhani', sans-serif; }
.metric-box .metric-label { font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

.pipeline-steps { padding: 4px 0; }
.pipeline-step { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #F0F0F0; font-size: 14px; color: #333; }
.pipeline-step:last-child { border-bottom: none; }
.step-num { background: #B31B1B; color: white; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; min-width: 22px; }
.step-icon { font-size: 18px; min-width: 24px; }
.step-text { flex: 1; line-height: 1.5; }

.gde-container { background: #0D1117; border-radius: 10px; padding: 24px 20px; margin: 12px 0; overflow-x: auto; box-shadow: inset 0 2px 8px rgba(0,0,0,0.4); }
.gde-flow { display: flex; align-items: center; gap: 0; min-width: max-content; padding: 8px 0; }
.gde-node { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.gde-node-box { border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 120px; transition: all 0.3s ease; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 8px rgba(255,214,0,0.3); } 50% { box-shadow: 0 0 20px rgba(255,214,0,0.7); } }
.gde-node-title { font-size: 11px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
.gde-node-sub { font-size: 10px; opacity: 0.7; margin-top: 2px; }
.gde-node-count { font-size: 14px; font-weight: 700; font-family: 'Rajdhani', sans-serif; margin-top: 4px; }
.gde-node-label { font-size: 10px; color: #666; text-align: center; max-width: 130px; }
.gde-arrow { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; padding: 0 6px; min-width: 70px; }
.gde-count-label { font-size: 10px; white-space: nowrap; text-align: center; }
.gde-legend { display: flex; gap: 20px; margin-top: 16px; flex-wrap: wrap; }
.gde-legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #888; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

.epic-card { background: linear-gradient(135deg, #B31B1B 0%, #7a1212 100%); border-radius: 10px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 4px 15px rgba(179,27,27,0.3); }
.epic-title { color: #FFC72C; font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }
.epic-value { color: rgba(255,255,255,0.85); font-size: 13px; margin-top: 6px; line-height: 1.6; }
.epic-meta { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.epic-badge { background: rgba(255,199,44,0.2); border: 1px solid rgba(255,199,44,0.5); color: #FFC72C; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }

.story-card { background: white; border: 1px solid #E8E8E8; border-left: 4px solid #B31B1B; border-radius: 8px; padding: 16px 20px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.story-id { font-size: 11px; color: #999; font-weight: 600; letter-spacing: 0.5px; }
.story-title { font-size: 14px; font-weight: 600; color: #1a1a1a; margin: 4px 0 8px 0; line-height: 1.4; }
.story-desc { font-size: 13px; color: #555; line-height: 1.6; font-style: italic; }
.story-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; align-items: center; }
.badge-priority-critical { background: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
.badge-priority-high { background: #FFF3E0; color: #E65100; border: 1px solid #FFCC80; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
.badge-priority-medium { background: #FFFDE7; color: #F57F17; border: 1px solid #FFF176; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
.badge-priority-low { background: #F5F5F5; color: #616161; border: 1px solid #E0E0E0; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
.badge-points { background: #FFC72C; color: #1a1a1a; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 700; }
.badge-sprint { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-type { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }

.ac-section { margin-top: 12px; }
.ac-title { font-size: 11px; font-weight: 700; color: #B31B1B; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.ac-item { font-size: 12px; color: #444; padding: 4px 0 4px 12px; border-left: 2px solid #FFC72C; margin: 4px 0; line-height: 1.5; }

.subtask-section { margin-top: 10px; }
.subtask-title { font-size: 11px; font-weight: 700; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.subtask-item { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #555; padding: 3px 0; }
.subtask-hrs { font-size: 10px; color: #999; background: #F5F5F5; padding: 1px 6px; border-radius: 8px; }

.risk-card { background: #FFF8E1; border: 1px solid #FFE082; border-left: 4px solid #FFC72C; border-radius: 8px; padding: 14px 18px; margin: 10px 0; }
.risk-title { font-size: 13px; font-weight: 700; color: #E65100; margin-bottom: 6px; }
.risk-item { font-size: 12px; color: #555; padding: 3px 0 3px 12px; border-left: 2px solid #FFB300; margin: 3px 0; }

.dod-card { background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 4px solid #2E7D32; border-radius: 8px; padding: 14px 18px; margin: 10px 0; }
.dod-title { font-size: 13px; font-weight: 700; color: #1B5E20; margin-bottom: 6px; }
.dod-item { font-size: 12px; color: #2E7D32; padding: 3px 0 3px 12px; border-left: 2px solid #66BB6A; margin: 3px 0; }

.jira-metrics { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
.jira-metric-box { background: white; border: 1px solid #E8E8E8; border-top: 3px solid #B31B1B; border-radius: 8px; padding: 12px 16px; min-width: 100px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); flex: 1; }
.jira-metric-value { font-size: 24px; font-weight: 700; color: #B31B1B; font-family: 'Rajdhani', sans-serif; }
.jira-metric-label { font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: 0.8px; }

/* Edit mode styles */
.edit-banner { background: linear-gradient(135deg, #1565C0, #0d47a1); border-radius: 8px; padding: 12px 18px; margin: 10px 0; display: flex; align-items: center; gap: 10px; }
.edit-banner-text { color: white; font-size: 13px; font-weight: 600; }
.edit-badge { background: rgba(255,199,44,0.2); border: 1px solid rgba(255,199,44,0.5); color: #FFC72C; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }

/* Jira push panel */
.jira-push-panel { background: linear-gradient(135deg, #0052CC 0%, #003d99 100%); border-radius: 10px; padding: 20px 24px; margin: 16px 0; box-shadow: 0 4px 15px rgba(0,82,204,0.3); }
.jira-push-title { color: #FFC72C; font-family: 'Rajdhani', sans-serif; font-size: 20px; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 6px; }
.jira-push-sub { color: rgba(255,255,255,0.75); font-size: 12px; }

.push-success { background: #E8F5E9; border: 1px solid #A5D6A7; border-left: 4px solid #2E7D32; border-radius: 8px; padding: 14px 18px; margin: 10px 0; }
.push-error { background: #FFEBEE; border: 1px solid #EF9A9A; border-left: 4px solid #C62828; border-radius: 8px; padding: 14px 18px; margin: 10px 0; }
.push-result-item { font-size: 12px; padding: 3px 0; }

.edited-badge { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="built-by-banner">
    <span class="byline">Built by</span>
    <span class="dot"></span>
    <span class="author">PRADEEP</span>
    <span class="dot"></span>
    <span class="byline">Enterprise AI</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div>
        <div class="header-sub">AI POWERED &nbsp;·&nbsp; LLAMA 3.3 &nbsp;·&nbsp; PANDAS</div>
        <h1>⚡ Enterprise AI Transformation &amp; Delivery Platform</h1>
    </div>
    <div class="version-badge">v3.1</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# GROQ SETUP
# -----------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []
if "jira_result" not in st.session_state:
    st.session_state.jira_result = None
if "edited_stories" not in st.session_state:
    st.session_state.edited_stories = {}
if "push_results" not in st.session_state:
    st.session_state.push_results = []


def extract_code(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


def make_gde_html(dataframes, file_names, code, result_df, state,
                  read_count=0, transform_count=0, out_count=0):
    aliases = list(dataframes.keys())
    real_aliases = [a for a in aliases if a != "df"]
    if not real_aliases:
        real_aliases = list(aliases)[:1] or ["df"]
    has_join = len(real_aliases) >= 2
    code_lower = (code or "").lower()
    trans_ops = []
    if "merge" in code_lower or "join" in code_lower: trans_ops.append("JOIN")
    if "groupby" in code_lower and "rank" in code_lower: trans_ops.append("RANK")
    if "pd.cut" in code_lower or "pd.qcut" in code_lower: trans_ops.append("BUCKET")
    if "re.sub" in code_lower or "replace" in code_lower: trans_ops.append("CLEAN")
    if "fillna" in code_lower: trans_ops.append("FILLNA")
    if not trans_ops: trans_ops.append("TRANSFORM")
    trans_label = " · ".join(trans_ops[:3])
    primary_rows = dataframes[real_aliases[0]].shape[0]
    secondary_rows = dataframes[real_aliases[1]].shape[0] if has_join else 0
    fname1 = file_names[0] if file_names else "file1.csv"
    fname2 = file_names[1] if len(file_names) > 1 else ""
    out_rows = len(result_df) if state == "done" and result_df is not None else out_count
    out_cols = len(result_df.columns) if state == "done" and result_df is not None else 0
    arrow1_color = "#29B6F6" if state in ("transforming","done") else ("#FFD600" if state == "reading" else "#444")
    arrow2_color = "#29B6F6" if state == "done" else "#444"
    input_border = "#1E90FF" if state in ("reading","transforming","done") else "#333"
    input_bg = "#1a2744" if state in ("reading","transforming","done") else "#111"
    input_color = "#7BB8FF" if state in ("reading","transforming","done") else "#444"
    if state == "transforming":
        trans_border="#FFD600"; trans_bg="#2a2a0a"; trans_color="#FFD600"; trans_anim="animation: pulse 1s infinite;"
    elif state == "done":
        trans_border="#29B6F6"; trans_bg="#0d2137"; trans_color="#29B6F6"; trans_anim=""
    else:
        trans_border="#333"; trans_bg="#111"; trans_color="#444"; trans_anim=""
    if state == "done":
        out_border="#29B6F6"; out_bg="#0d2137"; out_color="#29B6F6"
    else:
        out_border="#AB47BC"; out_bg="#1a1a2e"; out_color="#CE93D8"
    trans_status = "🟡 RUNNING" if state == "transforming" else ("🔵 COMPLETE" if state == "done" else "⏳ WAITING")
    in_count_display = f"{primary_rows:,}" if state in ("reading","transforming","done") else "–"
    in2_count_display = f"{secondary_rows:,}" if has_join and state in ("reading","transforming","done") else "–"
    tr_count_display = f"{primary_rows+secondary_rows:,} rec" if has_join and state in ("transforming","done") else (f"{primary_rows:,} rec" if state in ("transforming","done") else "–")
    out_count_display = f"{out_rows:,}" if state == "done" else "–"

    def svg_arrow(color, label=""):
        uid = abs(hash(label + color)) % 99999
        return f"""<div class="gde-arrow"><svg width="70" height="18" viewBox="0 0 70 18" xmlns="http://www.w3.org/2000/svg"><defs><marker id="ah{uid}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><polygon points="0 0, 6 3, 0 6" fill="{color}" /></marker></defs><line x1="2" y1="9" x2="62" y2="9" stroke="{color}" stroke-width="2.5" marker-end="url(#ah{uid})" /></svg><div class="gde-count-label" style="color:{color};">{label}</div></div>"""

    html = '<div class="gde-flow">'
    if has_join:
        html += f"""<div class="gde-node"><div style="display:flex;flex-direction:column;gap:10px;"><div class="gde-node"><div class="gde-node-box" style="background:{input_bg};border:2px solid {input_border};color:{input_color};min-width:120px;border-radius:8px;padding:10px 14px;text-align:center;"><div class="gde-node-title">📂 INPUT 1</div><div class="gde-node-sub">{fname1}</div><div class="gde-node-count">{in_count_display} rows</div></div><div class="gde-node-label">{real_aliases[0]}</div></div><div class="gde-node"><div class="gde-node-box" style="background:{input_bg};border:2px solid {input_border};color:{input_color};min-width:120px;border-radius:8px;padding:10px 14px;text-align:center;"><div class="gde-node-title">📂 INPUT 2</div><div class="gde-node-sub">{fname2}</div><div class="gde-node-count">{in2_count_display} rows</div></div><div class="gde-node-label">{real_aliases[1]}</div></div></div></div>"""
        html += svg_arrow(arrow1_color, f"{primary_rows+secondary_rows:,} in" if state in ("transforming","done") else "")
    else:
        html += f"""<div class="gde-node"><div class="gde-node-box" style="background:{input_bg};border:2px solid {input_border};color:{input_color};min-width:120px;border-radius:8px;padding:10px 14px;text-align:center;"><div class="gde-node-title">📂 INPUT</div><div class="gde-node-sub">{fname1}</div><div class="gde-node-count">{in_count_display} rows</div></div><div class="gde-node-label">{real_aliases[0]}</div></div>"""
        html += svg_arrow(arrow1_color, f"{primary_rows:,} in" if state in ("transforming","done") else "")
    html += f"""<div class="gde-node"><div class="gde-node-box" style="background:{trans_bg};border:2px solid {trans_border};color:{trans_color};min-width:130px;border-radius:8px;padding:10px 14px;text-align:center;{trans_anim}"><div class="gde-node-title">⚙ {trans_label}</div><div class="gde-node-sub">AI GENERATED</div><div class="gde-node-count">{tr_count_display}</div></div><div class="gde-node-label">{trans_status}</div></div>"""
    html += svg_arrow(arrow2_color, f"{out_count_display} out" if state == "done" else "")
    html += f"""<div class="gde-node"><div class="gde-node-box" style="background:{out_bg};border:2px solid {out_border};color:{out_color};min-width:120px;border-radius:8px;padding:10px 14px;text-align:center;"><div class="gde-node-title">💾 OUTPUT</div><div class="gde-node-sub">{out_cols} columns</div><div class="gde-node-count">{out_count_display} rows</div></div><div class="gde-node-label">RESULT</div></div></div>"""
    legend = """<div class="gde-legend"><div class="gde-legend-item"><div class="legend-dot" style="background:#1E90FF;"></div> Input</div><div class="gde-legend-item"><div class="legend-dot" style="background:#FFD600;"></div> Running</div><div class="gde-legend-item"><div class="legend-dot" style="background:#29B6F6;"></div> Complete</div><div class="gde-legend-item"><div class="legend-dot" style="background:#AB47BC;"></div> Output</div></div>"""
    return f'<div class="gde-container">{html}{legend}</div>'


def build_pipeline_log(code, dataframes, result_df, file_names, original_rows):
    aliases = list(dataframes.keys())
    summary_prompt = f"""You are a data pipeline narrator for a business audience.
Code: ```python\n{code}\n```
Input files: {file_names}, Rows before: {original_rows}, Rows after: {len(result_df)}, Columns: {result_df.columns.tolist()}
Describe in 4-8 plain-English bullet steps. Each step starts with: Loaded, Joined, Cleaned, Computed, Filtered, Sorted, Selected.
Return ONLY a JSON array of strings: ["Step one", "Step two"]. No markdown."""
    try:
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":summary_prompt}], temperature=0.2)
        raw = resp.choices[0].message.content.strip()
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        steps = json.loads(arr_match.group()) if arr_match else [raw]
    except:
        steps = [f"Loaded {original_rows:,} rows from {', '.join(file_names)}", "Applied AI-generated transformations", f"Produced {len(result_df):,} rows × {len(result_df.columns)} columns"]
    icon_map = {"load":"📂","read":"📂","join":"🔗","merge":"🔗","combined":"🔗","clean":"🧹","strip":"🧹","remov":"🧹","replac":"🧹","comput":"⚙️","calculat":"⚙️","creat":"⚙️","add":"⚙️","generat":"⚙️","filter":"🔍","kept":"🔍","exclud":"🔍","select":"🔍","sort":"↕️","order":"↕️","rank":"🏅","format":"✏️"}
    def pick_icon(t):
        tl = t.lower()
        for kw, icon in icon_map.items():
            if kw in tl: return icon
        return "✅"
    steps_html = '<div class="pipeline-steps">'
    for i, step in enumerate(steps, 1):
        steps_html += f'<div class="pipeline-step"><span class="step-num">{i}</span><span class="step-icon">{pick_icon(step)}</span><span class="step-text">{step}</span></div>'
    steps_html += "</div>"
    new_cols = [c for c in result_df.columns if c not in list(dataframes.values())[0].columns]
    n_joins = max(0, len([a for a in aliases if a != "df"]) - 1)
    metrics_html = f"""<div class="metric-row"><div class="metric-box"><div class="metric-value">{len(file_names)}</div><div class="metric-label">Files</div></div><div class="metric-box"><div class="metric-value">{original_rows:,}</div><div class="metric-label">Rows In</div></div><div class="metric-box"><div class="metric-value">{len(result_df):,}</div><div class="metric-label">Rows Out</div></div><div class="metric-box"><div class="metric-value">{len(result_df.columns)}</div><div class="metric-label">Columns</div></div><div class="metric-box"><div class="metric-value">{len(new_cols)}</div><div class="metric-label">New Cols</div></div><div class="metric-box"><div class="metric-value">{n_joins}</div><div class="metric-label">Joins</div></div></div>"""
    return metrics_html, steps_html


def safe_exec_multi(dataframes: dict, code: str) -> pd.DataFrame:
    code = extract_code(code)
    exec_globals = {**globals(), **dataframes}
    try:
        exec(compile(code, "<ai_etl>", "exec"), exec_globals)
    except Exception as exc:
        raise RuntimeError(f"Execution failed: {exc}\n\nCode:\n{code}") from exc
    primary = "df" if "df" in dataframes else list(dataframes.keys())[0]
    output = exec_globals.get("result", exec_globals.get(primary, list(dataframes.values())[0]))
    if not isinstance(output, pd.DataFrame):
        raise RuntimeError(f"AI produced {type(output).__name__} instead of DataFrame.")
    return output


def build_system_prompt(dataframes: dict) -> str:
    schema_lines = ""
    for alias, df in dataframes.items():
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        schema_lines += f"\n  {alias}: columns={df.columns.tolist()}, dtypes={dtypes}, shape={df.shape}"
    aliases = list(dataframes.keys())
    primary = aliases[0] if aliases else "df"
    join_examples = ""
    if len(aliases) >= 2:
        a, b = aliases[0], aliases[1]
        common = list(set(dataframes[a].columns) & set(dataframes[b].columns))
        jcol = common[0] if common else "id"
        join_examples = f"\nresult = pd.merge({a}, {b}, on='{jcol}', how='inner')"
    return f"""You are a Senior Enterprise Data Engineer.
AVAILABLE DATAFRAMES:{schema_lines}
PRIMARY DATAFRAME: '{primary}'
RULES: Use ONLY listed aliases. Store output in 'result'. Handle nulls. Strip whitespace. Use vectorised ops. Return ONLY executable Python.{join_examples}"""


PROJECT_TYPE_PROMPTS = {
    "🌐 Web Application": "You are a Senior Agile Delivery Manager specialising in Web Application delivery.",
    "📱 Mobile App": "You are a Senior Agile Delivery Manager specialising in Mobile Application delivery.",
    "📊 Data / ETL Pipeline": "You are a Senior Agile Delivery Manager specialising in Data Engineering and ETL pipelines.",
    "🔗 API / Integration": "You are a Senior Agile Delivery Manager specialising in API and Systems Integration.",
    "☁️ Cloud / Infrastructure": "You are a Senior Agile Delivery Manager specialising in Cloud Infrastructure and DevOps.",
    "🔒 Security Feature": "You are a Senior Agile Delivery Manager specialising in Cybersecurity features.",
    "🤖 AI / ML Feature": "You are a Senior Agile Delivery Manager specialising in AI and Machine Learning product delivery.",
    "📋 General / Other": "You are a Senior Agile Delivery Manager with 15+ years enterprise software delivery experience.",
}

def build_jira_prompt(description, project_type, team_size, sprint_length, methodology):
    system = PROJECT_TYPE_PROMPTS.get(project_type, PROJECT_TYPE_PROMPTS["📋 General / Other"])
    user = f"""BUSINESS REQUIREMENT:\n{description}\n\nTEAM CONTEXT: Project Type: {project_type}, Team Size: {team_size}, Sprint: {sprint_length} weeks, Methodology: {methodology}\n\nGenerate complete Jira breakdown. Return ONLY valid JSON:\n{{"epic":{{"title":"","business_value":"","objective":"","estimated_sprints":3,"definition_of_done":[]}},"stories":[{{"id":"US-001","title":"","user_story":"As a [role], I want [feature], so that [benefit]","priority":"High","story_points":5,"sprint":"Sprint 1","type":"Feature","acceptance_criteria":["Given..."],"subtasks":[{{"title":"","hours":4}}]}}],"risks":[{{"title":"","description":""}}],"dependencies":[]}}\n\nRULES: 4-7 stories, Fibonacci points, Gherkin AC, return ONLY JSON."""
    return system, user


def render_jira_cards(data):
    epic = data.get("epic", {})
    stories = data.get("stories", [])
    risks = data.get("risks", [])
    dependencies = data.get("dependencies", [])
    total_points = sum(s.get("story_points", 0) for s in stories)
    sprints_needed = epic.get("estimated_sprints", "?")
    html = f"""<div class="jira-metrics"><div class="jira-metric-box"><div class="jira-metric-value">{len(stories)}</div><div class="jira-metric-label">Stories</div></div><div class="jira-metric-box"><div class="jira-metric-value">{total_points}</div><div class="jira-metric-label">Total Points</div></div><div class="jira-metric-box"><div class="jira-metric-value">{sprints_needed}</div><div class="jira-metric-label">Sprints</div></div><div class="jira-metric-box"><div class="jira-metric-value">{len(risks)}</div><div class="jira-metric-label">Risks</div></div><div class="jira-metric-box"><div class="jira-metric-value">{len(dependencies)}</div><div class="jira-metric-label">Dependencies</div></div></div>"""
    dod_items = "".join(f'<div class="dod-item">✓ {d}</div>' for d in epic.get("definition_of_done", []))
    html += f"""<div class="epic-card"><div class="epic-title">🏆 EPIC: {epic.get('title','')}</div><div class="epic-value">{epic.get('business_value','')}</div><div class="epic-value" style="margin-top:6px;"><b>Objective:</b> {epic.get('objective','')}</div><div class="epic-meta"><span class="epic-badge">📅 {sprints_needed} Sprints</span><span class="epic-badge">📊 {total_points} Story Points</span><span class="epic-badge">📝 {len(stories)} Stories</span></div></div>"""
    if epic.get("definition_of_done"):
        html += f'<div class="dod-card"><div class="dod-title">✅ Definition of Done</div>{dod_items}</div>'
    return html, stories, risks, dependencies


# ============================================================
# NEW: Jira Push Functions
# ============================================================

def get_jira_priority_id(priority_name: str) -> str:
    """Map priority name to Jira priority name."""
    mapping = {
        "critical": "Highest",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return mapping.get(priority_name.lower(), "Medium")


def get_jira_issue_type(type_name: str) -> str:
    """Map story type to Jira issue type."""
    mapping = {
        "feature": "Story",
        "bug": "Bug",
        "task": "Task",
        "spike": "Story",
        "epic": "Epic",
    }
    return mapping.get(type_name.lower(), "Story")


def push_story_to_jira(jira_url: str, email: str, api_token: str, project_key: str, story: dict, epic_link: str = None) -> dict:
    """Push a single story to Jira Cloud and return result."""
    jira_url = jira_url.rstrip("/")
    auth = b64encode(f"{email}:{api_token}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Build description in Atlassian Document Format (ADF)
    ac_list = story.get("acceptance_criteria", [])
    subtasks_list = story.get("subtasks", [])

    ac_content = []
    for ac in ac_list:
        ac_content.append({
            "type": "listItem",
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": ac}]}]
        })

    sub_content = []
    for sub in subtasks_list:
        sub_content.append({
            "type": "listItem",
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"{sub.get('title','')} (~{sub.get('hours',0)}h)"}]}]
        })

    description_doc = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": story.get("user_story", ""), "marks": [{"type": "em"}]}]
            },
            {
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": "Acceptance Criteria"}]
            },
            {
                "type": "bulletList",
                "content": ac_content if ac_content else [{"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "N/A"}]}]}]
            },
        ]
    }

    if sub_content:
        description_doc["content"].append({
            "type": "heading",
            "attrs": {"level": 3},
            "content": [{"type": "text", "text": "Subtasks / Tasks"}]
        })
        description_doc["content"].append({
            "type": "bulletList",
            "content": sub_content
        })

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": story.get("title", "Untitled Story"),
            "description": description_doc,
            "issuetype": {"name": get_jira_issue_type(story.get("type", "Feature"))},
            "priority": {"name": get_jira_priority_id(story.get("priority", "Medium"))},
        }
    }

    # Add story points if supported
    story_points = story.get("story_points")
    if story_points:
        payload["fields"]["story_points"] = story_points

    # Add labels
    payload["fields"]["labels"] = [
        story.get("sprint", "Sprint-1").replace(" ", "-"),
        "AI-Generated",
        story.get("type", "Feature").replace(" ", "-"),
    ]

    try:
        resp = requests.post(
            f"{jira_url}/rest/api/3/issue",
            headers=headers,
            json=payload,
            timeout=15
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            return {
                "success": True,
                "issue_key": data.get("key", "???"),
                "issue_url": f"{jira_url}/browse/{data.get('key','')}",
                "title": story.get("title", ""),
                "story_id": story.get("id", ""),
            }
        else:
            err_data = resp.json() if resp.content else {}
            err_msg = err_data.get("errorMessages", []) or list(err_data.get("errors", {}).values())
            return {
                "success": False,
                "title": story.get("title", ""),
                "story_id": story.get("id", ""),
                "error": f"HTTP {resp.status_code}: {', '.join(err_msg) if err_msg else resp.text[:200]}"
            }
    except requests.exceptions.ConnectionError:
        return {"success": False, "title": story.get("title",""), "story_id": story.get("id",""), "error": "Cannot connect to Jira. Check URL."}
    except requests.exceptions.Timeout:
        return {"success": False, "title": story.get("title",""), "story_id": story.get("id",""), "error": "Request timed out."}
    except Exception as e:
        return {"success": False, "title": story.get("title",""), "story_id": story.get("id",""), "error": str(e)}


def test_jira_connection(jira_url: str, email: str, api_token: str, project_key: str) -> dict:
    """Test Jira connection and project access."""
    jira_url = jira_url.rstrip("/")
    auth = b64encode(f"{email}:{api_token}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}
    try:
        # Test project access
        resp = requests.get(f"{jira_url}/rest/api/3/project/{project_key}", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "project_name": data.get("name",""), "project_key": data.get("key","")}
        elif resp.status_code == 401:
            return {"success": False, "error": "Authentication failed. Check email and API token."}
        elif resp.status_code == 404:
            return {"success": False, "error": f"Project '{project_key}' not found. Check project key."}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot reach Jira. Check the URL (e.g. https://yourorg.atlassian.net)."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["⚡ AI ETL Engine", "📋 AI Jira Breakdown", "🎬 Demo & Benefits"])


# ============================================================
# TAB 1 — AI ETL ENGINE
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Business Description</div>', unsafe_allow_html=True)
    etl_prompt = st.text_area("Describe data transformation", key="etl_prompt", height=140)
    uploaded_files = st.file_uploader("Upload CSV File(s) — upload multiple files to enable joins", type=["csv"], accept_multiple_files=True, key="etl_upload")

    if uploaded_files:
        st.markdown('<div class="section-title">Uploaded Files Preview</div>', unsafe_allow_html=True)
        _preview_dfs = {}
        for i, f in enumerate(uploaded_files):
            alias = f"df{i+1}" if len(uploaded_files) > 1 else "df"
            _df = pd.read_csv(f)
            _preview_dfs[alias] = _df
            with st.expander(f"📄 {f.name}  →  alias: `{alias}`  |  {_df.shape[0]:,} rows × {_df.shape[1]} cols"):
                st.dataframe(_df.head(5), use_container_width=True)
        if len(uploaded_files) > 1:
            st.info(f"**{len(uploaded_files)} files loaded.** Reference them as: {', '.join(f'`{a}`' for a in _preview_dfs)}.")

    if st.button("▶ Execute ETL", key="run_etl"):
        if not etl_prompt.strip():
            st.warning("Enter a transformation description."); st.stop()
        if not uploaded_files:
            st.warning("Upload at least one CSV file."); st.stop()

        dataframes = {}
        for i, f in enumerate(uploaded_files):
            f.seek(0)
            alias = f"df{i+1}" if len(uploaded_files) > 1 else "df"
            dataframes[alias] = pd.read_csv(f)
        if len(uploaded_files) == 1:
            dataframes["df"] = list(dataframes.values())[0]

        primary_alias = "df" if len(uploaded_files) == 1 else "df1"
        original_rows = dataframes[primary_alias].shape[0]
        system_prompt = build_system_prompt(dataframes)
        file_names = [f.name for f in uploaded_files]

        st.markdown('<div class="section-title">⚡ Execution Flow</div>', unsafe_allow_html=True)
        gde_slot = st.empty()
        gde_slot.markdown(make_gde_html(dataframes, file_names, "", None, "reading"), unsafe_allow_html=True)
        time.sleep(0.6)

        MAX_ATTEMPTS = 3; ai_code = ""; transformed_df = None; last_error = None
        conversation = [{"role":"system","content":system_prompt},{"role":"user","content":etl_prompt}]
        gde_slot.markdown(make_gde_html(dataframes, file_names, "", None, "transforming"), unsafe_allow_html=True)

        with st.spinner("⚙️ AI is generating and executing pipeline..."):
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if last_error and attempt > 1:
                    conversation.append({"role":"assistant","content":ai_code})
                    conversation.append({"role":"user","content":f"Attempt {attempt-1} raised:\n{last_error}\n\nFix: no markdown fences, store result in 'result'."})
                response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=conversation, temperature=0.1)
                ai_code = response.choices[0].message.content
                try:
                    transformed_df = safe_exec_multi(dataframes, ai_code); last_error = None; break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt == MAX_ATTEMPTS:
                        st.error(f"ETL failed after {MAX_ATTEMPTS} attempts.\n\n{exc}")
                        transformed_df = list(dataframes.values())[0].copy()

        gde_slot.markdown(make_gde_html(dataframes, file_names, extract_code(ai_code), transformed_df, "done"), unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Pipeline Execution Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating plain-English pipeline summary..."):
            metrics_html, steps_html = build_pipeline_log(extract_code(ai_code), dataframes, transformed_df, file_names, original_rows)
        st.markdown(metrics_html, unsafe_allow_html=True)
        with st.expander("📋 View detailed pipeline steps", expanded=False):
            st.markdown(steps_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Transformed Output</div>', unsafe_allow_html=True)
        total_rows = len(transformed_df)
        col_info, col_select = st.columns([3, 1])
        col_info.markdown(f"<span style='font-size:13px;color:#666;'>Total records: <b>{total_rows:,}</b></span>", unsafe_allow_html=True)
        display_options = sorted(set(n for n in [20,50,100,500,1000,total_rows] if n <= total_rows)) or [total_rows]
        display_n = col_select.selectbox("Show rows", options=display_options, index=0, key="display_rows")
        st.dataframe(transformed_df.head(display_n), use_container_width=True)

        st.session_state.history.append({"Time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Prompt":etl_prompt,"Files":", ".join(file_names),"Rows Before":original_rows,"Rows After":len(transformed_df),"Status":"OK" if last_error is None else "FAILED"})

        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.download_button("⬇ Download CSV", transformed_df.to_csv(index=False).encode("utf-8"), "etl_output.csv", "text/csv")
        xlsx_buf = BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            transformed_df.to_excel(writer, sheet_name="Transformed_Data", index=False)
            pd.DataFrame(st.session_state.history).to_excel(writer, sheet_name="Audit_Log", index=False)
        col2.download_button("⬇ Download Excel", xlsx_buf.getvalue(), "etl_output.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================
# TAB 2 — AI JIRA BREAKDOWN (ENHANCED)
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Project Configuration</div>', unsafe_allow_html=True)
    project_type = st.selectbox("Project Type", list(PROJECT_TYPE_PROMPTS.keys()), key="proj_type")
    col_a, col_b, col_c = st.columns(3)
    team_size = col_a.selectbox("Team Size", [2,3,4,5,6,8,10,12,15,20], index=3, key="team_size")
    sprint_length = col_b.selectbox("Sprint Length (weeks)", [1,2,3], index=1, key="sprint_len")
    methodology = col_c.selectbox("Methodology", ["Scrum","Kanban","SAFe","Scrumban"], key="methodology")

    st.markdown('<div class="section-title">Business Requirement</div>', unsafe_allow_html=True)
    jira_prompt = st.text_area("Describe the feature, initiative or product requirement in detail", key="jira_prompt", height=160, placeholder="Example: Build a customer self-service portal where users can view invoices, raise support tickets, track order status, and manage their account profile. Must support SSO login and mobile-responsive design.")

    if st.button("🚀 Generate Jira Breakdown", key="run_jira"):
        if not jira_prompt.strip():
            st.warning("Enter a business requirement."); st.stop()
        with st.spinner(f"🧠 Generating {project_type} Jira breakdown using LLaMA 3.3 70B..."):
            sys_prompt, user_prompt = build_jira_prompt(jira_prompt, project_type, team_size, sprint_length, methodology)
            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}], temperature=0.3, max_tokens=4000)
            raw_output = response.choices[0].message.content.strip()
        try:
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            jira_data = json.loads(json_match.group()) if json_match else {}
        except:
            jira_data = {}
        if not jira_data:
            st.error("Could not parse structured output."); st.markdown(raw_output); st.stop()
        st.session_state.jira_result = {"data":jira_data,"raw":raw_output,"type":project_type}
        # Reset edited stories when new breakdown is generated
        st.session_state.edited_stories = {}
        st.session_state.push_results = []

    if st.session_state.jira_result:
        jira_data = st.session_state.jira_result["data"]
        project_type_display = st.session_state.jira_result["type"]

        st.markdown('<div class="section-title">📊 Breakdown Summary</div>', unsafe_allow_html=True)
        header_html, stories, risks, dependencies = render_jira_cards(jira_data)
        st.markdown(header_html, unsafe_allow_html=True)

        # ============================================================
        # FEATURE 1: EDIT USER STORIES
        # ============================================================
        st.markdown('<div class="section-title">📝 User Stories — Review & Edit</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="edit-banner">
            <div style="font-size:18px;">✏️</div>
            <div class="edit-banner-text">PO Edit Mode — Expand any story below to review and edit before pushing to Jira</div>
            <div class="edit-badge">EDITABLE</div>
        </div>
        """, unsafe_allow_html=True)

        priority_badge = {"critical":"badge-priority-critical","high":"badge-priority-high","medium":"badge-priority-medium","low":"badge-priority-low"}
        PRIORITY_OPTIONS = ["Critical", "High", "Medium", "Low"]
        SPRINT_OPTIONS = ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4", "Sprint 5", "Sprint 6"]
        TYPE_OPTIONS = ["Feature", "Bug", "Task", "Spike", "Technical Debt"]
        POINTS_OPTIONS = [1, 2, 3, 5, 8, 13, 21]

        for idx, story in enumerate(stories):
            story_id = story.get("id", f"US-{idx+1:03d}")
            pri = story.get("priority","Medium")
            pts = story.get("story_points", 0)
            stype = story.get("type","Feature")
            edited = story_id in st.session_state.edited_stories
            edit_label = " ✏️ [EDITED]" if edited else ""

            # Use edited data if available
            current = st.session_state.edited_stories.get(story_id, story)

            with st.expander(f"  {story_id} · {current.get('title','')}  [{current.get('priority', pri)}] [{current.get('story_points', pts)} pts]{edit_label}", expanded=False):

                # Two-column layout: view on left, edit on right
                view_col, edit_col = st.columns([1, 1])

                with view_col:
                    st.markdown("**📖 Current Version**")
                    cpri = current.get('priority', pri)
                    pbadge = priority_badge.get(cpri.lower(), "badge-priority-medium")
                    ac_items = "".join(f'<div class="ac-item">• {ac}</div>' for ac in current.get("acceptance_criteria",[]))
                    sub_items = "".join(f'<div class="subtask-item">☐ {s.get("title","")} <span class="subtask-hrs">~{s.get("hours",0)}h</span></div>' for s in current.get("subtasks",[]))
                    edited_badge_html = '<span class="edited-badge">✏️ EDITED</span>' if edited else ''
                    st.markdown(f"""<div class="story-card">
                        <div class="story-id">{story_id} &nbsp;·&nbsp; {project_type_display} &nbsp;{edited_badge_html}</div>
                        <div class="story-title">{current.get('title','')}</div>
                        <div class="story-desc">{current.get('user_story','')}</div>
                        <div class="story-badges">
                            <span class="{pbadge}">🔴 {cpri}</span>
                            <span class="badge-points">⭐ {current.get('story_points', pts)} pts</span>
                            <span class="badge-sprint">🏃 {current.get('sprint','')}</span>
                            <span class="badge-type">🏷 {current.get('type', stype)}</span>
                        </div>
                        <div class="ac-section"><div class="ac-title">✅ Acceptance Criteria</div>{ac_items}</div>
                        <div class="subtask-section"><div class="subtask-title">🔧 Subtasks</div>{sub_items}</div>
                    </div>""", unsafe_allow_html=True)

                with edit_col:
                    st.markdown("**✏️ Edit Fields**")

                    new_title = st.text_input(
                        "Story Title",
                        value=current.get("title", story.get("title","")),
                        key=f"title_{story_id}"
                    )
                    new_user_story = st.text_area(
                        "User Story (As a... I want... So that...)",
                        value=current.get("user_story", story.get("user_story","")),
                        key=f"us_{story_id}",
                        height=80
                    )

                    ec1, ec2, ec3 = st.columns(3)
                    new_priority = ec1.selectbox(
                        "Priority",
                        PRIORITY_OPTIONS,
                        index=PRIORITY_OPTIONS.index(current.get("priority", pri)) if current.get("priority", pri) in PRIORITY_OPTIONS else 1,
                        key=f"pri_{story_id}"
                    )
                    current_pts = current.get("story_points", pts)
                    pts_index = POINTS_OPTIONS.index(current_pts) if current_pts in POINTS_OPTIONS else 3
                    new_points = ec2.selectbox(
                        "Story Points",
                        POINTS_OPTIONS,
                        index=pts_index,
                        key=f"pts_{story_id}"
                    )
                    current_sprint = current.get("sprint", story.get("sprint","Sprint 1"))
                    sprint_index = SPRINT_OPTIONS.index(current_sprint) if current_sprint in SPRINT_OPTIONS else 0
                    new_sprint = ec3.selectbox(
                        "Sprint",
                        SPRINT_OPTIONS,
                        index=sprint_index,
                        key=f"spr_{story_id}"
                    )

                    current_type = current.get("type", stype)
                    type_index = TYPE_OPTIONS.index(current_type) if current_type in TYPE_OPTIONS else 0
                    new_type = st.selectbox(
                        "Issue Type",
                        TYPE_OPTIONS,
                        index=type_index,
                        key=f"typ_{story_id}"
                    )

                    st.markdown("**Acceptance Criteria** *(one per line)*")
                    existing_ac = "\n".join(current.get("acceptance_criteria", story.get("acceptance_criteria",[])))
                    new_ac_raw = st.text_area(
                        "Acceptance Criteria",
                        value=existing_ac,
                        key=f"ac_{story_id}",
                        height=100,
                        label_visibility="collapsed"
                    )

                    st.markdown("**Subtasks** *(format: Task title | hours)*")
                    existing_subs = "\n".join(
                        f"{s.get('title','')} | {s.get('hours',0)}"
                        for s in current.get("subtasks", story.get("subtasks",[]))
                    )
                    new_subs_raw = st.text_area(
                        "Subtasks",
                        value=existing_subs,
                        key=f"sub_{story_id}",
                        height=80,
                        label_visibility="collapsed"
                    )

                    save_col, reset_col = st.columns(2)
                    with save_col:
                        if st.button(f"💾 Save Changes", key=f"save_{story_id}"):
                            # Parse ACs
                            new_ac = [line.strip() for line in new_ac_raw.split("\n") if line.strip()]
                            # Parse subtasks
                            new_subs = []
                            for line in new_subs_raw.split("\n"):
                                line = line.strip()
                                if line:
                                    parts = line.split("|")
                                    title_part = parts[0].strip()
                                    hours_part = 0
                                    if len(parts) > 1:
                                        try:
                                            hours_part = int(parts[1].strip())
                                        except:
                                            hours_part = 0
                                    new_subs.append({"title": title_part, "hours": hours_part})

                            st.session_state.edited_stories[story_id] = {
                                **story,
                                "title": new_title,
                                "user_story": new_user_story,
                                "priority": new_priority,
                                "story_points": new_points,
                                "sprint": new_sprint,
                                "type": new_type,
                                "acceptance_criteria": new_ac,
                                "subtasks": new_subs,
                            }
                            st.success(f"✅ {story_id} saved! Changes will be reflected above.")
                            st.rerun()

                    with reset_col:
                        if edited and st.button(f"↩ Reset to AI", key=f"reset_{story_id}"):
                            if story_id in st.session_state.edited_stories:
                                del st.session_state.edited_stories[story_id]
                            st.success(f"↩ {story_id} reset to original AI version.")
                            st.rerun()

        # Show edit summary
        if st.session_state.edited_stories:
            edited_count = len(st.session_state.edited_stories)
            st.markdown(f"""
            <div style="background:#E8F5E9;border:1px solid #A5D6A7;border-left:4px solid #2E7D32;border-radius:8px;padding:12px 18px;margin:12px 0;">
                <strong style="color:#1B5E20;">✏️ {edited_count} story/stories edited by PO</strong>
                <span style="color:#555;font-size:12px;margin-left:8px;">— Edited stories will be pushed with your changes to Jira</span>
            </div>
            """, unsafe_allow_html=True)

        if risks:
            st.markdown('<div class="section-title">⚠️ Risks & Dependencies</div>', unsafe_allow_html=True)
            risk_html = '<div class="risk-card"><div class="risk-title">⚠️ Identified Risks</div>'
            for r in risks: risk_html += f'<div class="risk-item"><b>{r.get("title","")}</b> — {r.get("description","")}</div>'
            risk_html += "</div>"
            if dependencies:
                risk_html += '<div class="risk-card" style="border-left-color:#1E90FF;background:#E3F2FD;"><div class="risk-title" style="color:#1565C0;">🔗 Dependencies</div>'
                for d in dependencies: risk_html += f'<div class="risk-item" style="border-left-color:#1E90FF;">{d}</div>'
                risk_html += "</div>"
            st.markdown(risk_html, unsafe_allow_html=True)

        # ============================================================
        # FEATURE 2: PUSH TO JIRA
        # ============================================================
        st.markdown("---")
        st.markdown("""
        <div class="jira-push-panel">
            <div class="jira-push-title">🚀 Push Stories to Jira</div>
            <div class="jira-push-sub">Connect to your Jira Cloud instance and push all reviewed stories directly — edited versions take priority</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔌 Jira Connection Settings", expanded=True):
            st.markdown("""
            <div style="background:#FFF8E1;border:1px solid #FFE082;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#6D4C41;">
                💡 <strong>Where to find your API Token:</strong> Go to <code>id.atlassian.com → Security → API tokens → Create API token</code>. 
                Your Jira URL is typically <code>https://yourorg.atlassian.net</code>
            </div>
            """, unsafe_allow_html=True)

            push_col1, push_col2 = st.columns(2)
            jira_url_input = push_col1.text_input(
                "Jira URL",
                placeholder="https://yourorg.atlassian.net",
                key="jira_url_input"
            )
            jira_email_input = push_col2.text_input(
                "Jira Account Email",
                placeholder="you@yourorg.com",
                key="jira_email_input"
            )

            push_col3, push_col4 = st.columns(2)
            jira_token_input = push_col3.text_input(
                "API Token",
                type="password",
                placeholder="Your Jira API token",
                key="jira_token_input"
            )
            jira_project_input = push_col4.text_input(
                "Project Key",
                placeholder="e.g. PROJ, DEV, ENG",
                key="jira_project_input"
            )

            test_col, push_btn_col = st.columns([1, 2])

            with test_col:
                if st.button("🔍 Test Connection", key="test_jira_conn"):
                    if not all([jira_url_input, jira_email_input, jira_token_input, jira_project_input]):
                        st.warning("Fill in all fields to test connection.")
                    else:
                        with st.spinner("Testing connection..."):
                            result = test_jira_connection(jira_url_input, jira_email_input, jira_token_input, jira_project_input)
                        if result["success"]:
                            st.success(f"✅ Connected! Project: **{result.get('project_name','')}** ({result.get('project_key','')})")
                        else:
                            st.error(f"❌ {result['error']}")

            with push_btn_col:
                if st.button("🚀 Push All Stories to Jira", key="push_to_jira", type="primary"):
                    if not all([jira_url_input, jira_email_input, jira_token_input, jira_project_input]):
                        st.warning("Please fill in all Jira connection fields before pushing.")
                    else:
                        # Build final stories (edited override original)
                        final_stories = []
                        for s in stories:
                            sid = s.get("id","")
                            final_stories.append(st.session_state.edited_stories.get(sid, s))

                        push_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, story in enumerate(final_stories):
                            status_text.markdown(f"⏳ Pushing **{story.get('id','')}**: {story.get('title','')[:60]}...")
                            result = push_story_to_jira(
                                jira_url_input,
                                jira_email_input,
                                jira_token_input,
                                jira_project_input,
                                story
                            )
                            push_results.append(result)
                            progress_bar.progress((i + 1) / len(final_stories))
                            time.sleep(0.3)  # Rate limiting courtesy

                        status_text.empty()
                        st.session_state.push_results = push_results

                        success_count = sum(1 for r in push_results if r["success"])
                        fail_count = len(push_results) - success_count

                        if success_count > 0:
                            st.success(f"✅ **{success_count}/{len(push_results)} stories pushed successfully to Jira!**")
                        if fail_count > 0:
                            st.error(f"❌ **{fail_count} stories failed to push.** See details below.")

        # Show push results
        if st.session_state.push_results:
            st.markdown('<div class="section-title">📬 Push Results</div>', unsafe_allow_html=True)

            success_items = [r for r in st.session_state.push_results if r["success"]]
            fail_items = [r for r in st.session_state.push_results if not r["success"]]

            if success_items:
                success_html = '<div class="push-success"><div style="font-size:14px;font-weight:700;color:#1B5E20;margin-bottom:8px;">✅ Successfully Created in Jira</div>'
                for r in success_items:
                    success_html += f'''<div class="push-result-item" style="color:#2E7D32;padding:5px 0;border-bottom:1px solid #C8E6C9;">
                        <strong>{r.get("story_id","")}</strong> · {r.get("title","")[:60]}
                        &nbsp;→&nbsp; <a href="{r.get("issue_url","")}" target="_blank" style="color:#1565C0;font-weight:700;">{r.get("issue_key","")}</a>
                        <span style="font-size:10px;color:#888;margin-left:8px;">{r.get("issue_url","")}</span>
                    </div>'''
                success_html += '</div>'
                st.markdown(success_html, unsafe_allow_html=True)

            if fail_items:
                fail_html = '<div class="push-error"><div style="font-size:14px;font-weight:700;color:#C62828;margin-bottom:8px;">❌ Failed to Push</div>'
                for r in fail_items:
                    fail_html += f'<div class="push-result-item" style="color:#B71C1C;padding:5px 0;border-bottom:1px solid #FFCDD2;"><strong>{r.get("story_id","")}</strong> · {r.get("title","")[:50]} — <em>{r.get("error","Unknown error")}</em></div>'
                fail_html += '</div>'
                st.markdown(fail_html, unsafe_allow_html=True)

        # ============================================================
        # EXPORT SECTION
        # ============================================================
        st.markdown('<div class="section-title">Export Jira Output</div>', unsafe_allow_html=True)

        # Use edited stories for export too
        final_export_stories = []
        for s in stories:
            sid = s.get("id","")
            final_export_stories.append(st.session_state.edited_stories.get(sid, s))

        col1, col2, col3 = st.columns(3)
        txt_lines = [f"EPIC: {jira_data.get('epic',{}).get('title','')}\n"]
        for s in final_export_stories:
            txt_lines.extend([f"\n{s.get('id','')} — {s.get('title','')}", f"  {s.get('user_story','')}", f"  Priority: {s.get('priority','')} | Points: {s.get('story_points','')} | Sprint: {s.get('sprint','')}","  Acceptance Criteria:"] + [f"    - {ac}" for ac in s.get("acceptance_criteria",[])] + ["  Subtasks:"] + [f"    □ {sub.get('title','')} (~{sub.get('hours',0)}h)" for sub in s.get("subtasks",[])])
        col1.download_button("⬇ Download TXT", "\n".join(txt_lines), "jira_breakdown.txt", "text/plain")
        xlsx_buf = BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            pd.DataFrame([{"ID":s.get("id",""),"Title":s.get("title",""),"User Story":s.get("user_story",""),"Priority":s.get("priority",""),"Story Points":s.get("story_points",""),"Sprint":s.get("sprint",""),"Type":s.get("type",""),"Edited": s.get("id","") in st.session_state.edited_stories} for s in final_export_stories]).to_excel(writer, sheet_name="Stories_Summary", index=False)
            ac_rows = [{"Story ID":s.get("id",""),"Story Title":s.get("title",""),"Acceptance Criteria":ac} for s in final_export_stories for ac in s.get("acceptance_criteria",[])]
            if ac_rows: pd.DataFrame(ac_rows).to_excel(writer, sheet_name="Acceptance_Criteria", index=False)
            sub_rows = [{"Story ID":s.get("id",""),"Subtask":sub.get("title",""),"Estimated Hours":sub.get("hours",0)} for s in final_export_stories for sub in s.get("subtasks",[])]
            if sub_rows: pd.DataFrame(sub_rows).to_excel(writer, sheet_name="Subtasks", index=False)
            if risks: pd.DataFrame(risks).to_excel(writer, sheet_name="Risks", index=False)
            # Push results if any
            if st.session_state.push_results:
                pd.DataFrame(st.session_state.push_results).to_excel(writer, sheet_name="Jira_Push_Log", index=False)
        col2.download_button("⬇ Download Excel (Jira Import)", xlsx_buf.getvalue(), "jira_breakdown.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        col3.download_button("⬇ Download JSON", json.dumps({"epic": jira_data.get("epic",{}), "stories": final_export_stories, "risks": risks, "dependencies": dependencies}, indent=2), "jira_breakdown.json", "application/json")


# ============================================================
# TAB 3 — DEMO & BENEFITS
# ============================================================
with tab3:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#B31B1B,#7a1212);border-radius:12px;padding:28px 32px;margin-bottom:24px;box-shadow:0 4px 20px rgba(179,27,27,0.4);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <div style="background:rgba(255,199,44,0.2);border:1px solid rgba(255,199,44,0.4);padding:4px 14px;border-radius:20px;font-size:11px;font-weight:700;color:#FFC72C;letter-spacing:2px;">🔴 LIVE DEMO</div>
            <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);padding:4px 14px;border-radius:20px;font-size:11px;color:rgba(255,255,255,0.7);letter-spacing:1px;">🔊 NARRATED · 43 SECONDS</div>
        </div>
        <div style="color:#FFC72C;font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;margin-bottom:8px;">⚡ See the Platform in Action</div>
        <div style="color:rgba(255,255,255,0.85);font-size:14px;line-height:1.7;max-width:720px;">Watch how <b style="color:#FFC72C;">AI ETL Engine</b> and <b style="color:#FFC72C;">AI Jira Breakdown</b> transform hours of manual work into seconds — with real-time pipeline animation, auto-generated user stories, PO editing, and direct Jira push.</div>
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html("""<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0a;font-family:'Inter',sans-serif;}
.vw{background:#070f1e;border:1px solid #1a3060;border-radius:16px;overflow:hidden;box-shadow:0 0 60px rgba(179,27,27,0.2),0 30px 60px rgba(0,0,0,0.6);max-width:940px;margin:0 auto;position:relative;}
.vtb{background:#0d1a2e;border-bottom:1px solid #1a3060;padding:10px 18px;display:flex;align-items:center;gap:10px;}
.tl-row{display:flex;gap:5px;}.tl{width:10px;height:10px;border-radius:50%;}.tl-r{background:#ff5f57;}.tl-y{background:#febc2e;}.tl-g{background:#28c840;}
.vtitle{flex:1;text-align:center;font-family:'Space Mono',monospace;font-size:11px;color:#4a7ab5;letter-spacing:1px;}
.rec{display:flex;align-items:center;gap:5px;font-family:'Space Mono',monospace;font-size:10px;color:#ff4444;}
.rdot{width:7px;height:7px;background:#ff4444;border-radius:50%;animation:blink 1s infinite;}
.stage{height:400px;position:relative;overflow:hidden;background:linear-gradient(180deg,#04091a 0%,#030812 100%);}
.scene{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity 0.5s ease;padding:24px;}
.scene.active{opacity:1;}
.term{background:#050d1a;border:1px solid rgba(179,27,27,0.4);border-radius:12px;padding:20px;width:100%;max-width:660px;box-shadow:0 0 30px rgba(179,27,27,0.1);}
.thdr{font-size:10px;color:#B31B1B;letter-spacing:2px;margin-bottom:16px;font-family:'Space Mono',monospace;}
.tln{font-size:13px;line-height:1.8;margin-bottom:3px;font-family:'Space Mono',monospace;}
.pr{color:#FFC72C;}.cm{color:#e8f4ff;}.op{color:#6b8bb5;}.hl{color:#FFC72C;font-weight:700;}
.cur{display:inline-block;width:8px;height:14px;background:#FFC72C;animation:blink 1s infinite;vertical-align:middle;margin-left:2px;}
.jboard{width:100%;max-width:720px;}
.jhdr{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.jlogo{width:28px;height:28px;background:linear-gradient(135deg,#B31B1B,#7a1212);border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:13px;color:#FFC72C;font-weight:900;}
.jpn{font-family:'Orbitron',monospace;font-size:12px;color:#e8f4ff;}
.aib{margin-left:auto;background:rgba(255,199,44,0.1);border:1px solid rgba(255,199,44,0.35);color:#FFC72C;font-family:'Space Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;letter-spacing:1px;}
.jcols{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.chdr{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:#4a7ab5;text-transform:uppercase;margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid #1a3060;}
.tkt{background:#0a1628;border:1px solid #1a3060;border-radius:7px;padding:8px 10px;margin-bottom:6px;font-size:11px;}
.tkt.new{border-color:rgba(255,199,44,0.4);background:rgba(255,199,44,0.03);animation:sIn 0.4s ease forwards;}
.tid{color:#B31B1B;font-family:'Space Mono',monospace;font-size:9px;margin-bottom:3px;}
.ttl{color:#e8f4ff;margin-bottom:5px;line-height:1.3;}
.tmeta{display:flex;gap:5px;align-items:center;}
.tp{font-size:9px;padding:2px 5px;border-radius:4px;font-family:'Space Mono',monospace;}
.ph{background:rgba(179,27,27,0.15);color:#ff6b6b;border:1px solid rgba(179,27,27,0.3);}
.pm{background:rgba(255,199,44,0.12);color:#FFC72C;border:1px solid rgba(255,199,44,0.25);}
.pd{background:rgba(40,200,64,0.1);color:#69F0AE;border:1px solid rgba(40,200,64,0.2);}
.tav{width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#B31B1B,#7a1212);font-size:7px;display:flex;align-items:center;justify-content:center;color:#FFC72C;font-weight:700;margin-left:auto;}
/* Edit scene */
.edit-scene{width:100%;max-width:720px;}
.edit-hdr{background:linear-gradient(135deg,#1565C0,#0d47a1);border-radius:8px;padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:10px;}
.edit-hdr-title{color:white;font-weight:700;font-size:12px;}
.edit-badge{background:rgba(255,199,44,0.2);border:1px solid rgba(255,199,44,0.5);color:#FFC72C;font-size:9px;padding:2px 8px;border-radius:10px;font-weight:700;}
.edit-card{background:#0a1628;border:1px solid #1a3060;border-radius:8px;padding:12px;}
.edit-field{background:#050d1a;border:1px solid #1E3A5F;border-radius:5px;padding:6px 10px;margin-bottom:8px;font-family:'Space Mono',monospace;font-size:10px;color:#4a7ab5;}
.edit-field.active{border-color:#FFC72C;color:#FFC72C;}
.edit-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;}
.edit-select{background:#050d1a;border:1px solid #1E3A5F;border-radius:5px;padding:6px;font-family:'Space Mono',monospace;font-size:10px;color:#4a7ab5;}
.save-btn{background:linear-gradient(135deg,#2E7D32,#1B5E20);color:white;border:none;border-radius:5px;padding:7px 18px;font-size:11px;font-weight:700;cursor:pointer;width:100%;}
/* Push scene */
.push-scene{width:100%;max-width:720px;}
.push-panel{background:linear-gradient(135deg,#0052CC,#003d99);border-radius:10px;padding:16px 20px;margin-bottom:14px;}
.push-title{color:#FFC72C;font-family:'Orbitron',monospace;font-size:14px;font-weight:700;margin-bottom:4px;}
.push-fields{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;}
.push-field{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);border-radius:5px;padding:7px 10px;font-size:10px;color:rgba(255,255,255,0.8);font-family:'Space Mono',monospace;}
.push-field.filled{border-color:#FFC72C;color:white;}
.push-progress{margin-top:8px;}
.push-bar-bg{background:rgba(255,255,255,0.1);border-radius:4px;height:6px;overflow:hidden;}
.push-bar-fill{background:linear-gradient(90deg,#FFC72C,#69F0AE);height:100%;border-radius:4px;transition:width 0.4s ease;}
.push-items{display:flex;flex-direction:column;gap:6px;}
.push-item{background:#0a1628;border:1px solid #1a3060;border-radius:6px;padding:8px 12px;display:flex;align-items:center;gap:10px;font-size:11px;}
.push-item.ok{border-color:rgba(40,200,64,0.4);}
.push-item.pending{border-color:rgba(255,199,44,0.3);}
.pi-id{font-family:'Space Mono',monospace;font-size:9px;color:#B31B1B;}
.pi-title{color:#e8f4ff;flex:1;}
.pi-key{font-family:'Space Mono',monospace;font-size:10px;color:#69F0AE;font-weight:700;}
.pi-dot{width:7px;height:7px;border-radius:50%;}
.etlw{width:100%;max-width:700px;}
.phdr{display:flex;align-items:center;gap:10px;margin-bottom:18px;}
.ptitle{font-family:'Orbitron',monospace;font-size:12px;color:#e8f4ff;}
.pst{margin-left:auto;font-family:'Space Mono',monospace;font-size:10px;color:#69F0AE;display:flex;align-items:center;gap:5px;}
.pflow{display:flex;align-items:center;margin-bottom:14px;}
.pnode{flex:1;background:#0a1628;border:1px solid #1a3060;border-radius:9px;padding:10px 6px;text-align:center;transition:all 0.3s;}
.pnode.active{border-color:#FFC72C;box-shadow:0 0 15px rgba(255,199,44,0.2);}
.pnode.done{border-color:rgba(40,200,64,0.4);box-shadow:0 0 10px rgba(40,200,64,0.1);}
.ni{font-size:16px;margin-bottom:3px;}.nn{font-family:'Space Mono',monospace;font-size:8px;color:#4a7ab5;letter-spacing:1px;text-transform:uppercase;}.ns{font-size:9px;color:#4a7ab5;margin-top:2px;}
.parr{width:24px;text-align:center;color:#B31B1B;font-size:11px;flex-shrink:0;}
.pkts{display:flex;gap:6px;flex-wrap:wrap;}
.pkt{background:rgba(179,27,27,0.1);border:1px solid rgba(179,27,27,0.25);border-radius:5px;padding:4px 9px;font-family:'Space Mono',monospace;font-size:10px;color:#ff9999;animation:pIn 0.4s ease forwards;opacity:0;}
.mw{width:100%;max-width:700px;}
.mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px;}
.mc{background:#0a1628;border:1px solid #1a3060;border-radius:10px;padding:14px;text-align:center;}
.mv{font-family:'Orbitron',monospace;font-size:28px;font-weight:900;color:#FFC72C;line-height:1;margin-bottom:4px;}
.ml{font-family:'Space Mono',monospace;font-size:9px;color:#4a7ab5;letter-spacing:2px;text-transform:uppercase;}
.md{font-size:11px;color:#69F0AE;margin-top:3px;}
.ins{background:#0a1628;border:1px solid #1a3060;border-radius:9px;padding:12px 14px;display:flex;gap:12px;align-items:center;}
.idot{width:7px;height:7px;background:#69F0AE;border-radius:50%;box-shadow:0 0 6px #69F0AE;animation:blink 1.5s infinite;flex-shrink:0;}
.ilbl{font-family:'Space Mono',monospace;font-size:9px;color:#4a7ab5;white-space:nowrap;}
.itxt{font-size:11px;color:#e8f4ff;line-height:1.5;}
.capbar{background:rgba(0,0,0,0.9);border-top:1px solid rgba(179,27,27,0.25);padding:11px 18px;min-height:50px;display:flex;align-items:center;gap:10px;}
.cico{font-size:14px;flex-shrink:0;}.ctxt{font-size:13px;color:#d0e8ff;font-style:italic;transition:opacity 0.3s;line-height:1.5;}
.cspk{color:#FFC72C;font-style:normal;font-weight:700;font-size:11px;font-family:'Space Mono',monospace;letter-spacing:1px;margin-right:7px;}
.ctrls{background:#060e1d;border-top:1px solid #1a3060;padding:11px 18px;display:flex;align-items:center;gap:12px;}
.cb{background:none;border:none;color:#4a7ab5;cursor:pointer;font-size:14px;padding:3px;transition:color 0.2s;}
.cb:hover{color:#FFC72C;}
.pbtn{background:#B31B1B;color:white;width:32px;height:32px;border-radius:50%;font-size:12px;display:flex;align-items:center;justify-content:center;border:none;cursor:pointer;box-shadow:0 0 14px rgba(179,27,27,0.4);}
.pbtn:hover{background:#8E1414;}
.pgbar{flex:1;height:4px;background:rgba(255,255,255,0.1);border-radius:2px;cursor:pointer;position:relative;}
.pgfill{height:100%;background:linear-gradient(90deg,#B31B1B,#FFC72C);border-radius:2px;transition:width 0.5s linear;position:relative;}
.pgfill::after{content:'';position:absolute;right:-4px;top:50%;transform:translateY(-50%);width:9px;height:9px;background:#FFC72C;border-radius:50%;box-shadow:0 0 7px #FFC72C;}
.tdisp{font-family:'Space Mono',monospace;font-size:11px;color:#4a7ab5;white-space:nowrap;}
.sdots{display:flex;gap:5px;}
.sd{width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,0.15);cursor:pointer;transition:all 0.2s;}
.sd.active{background:#FFC72C;box-shadow:0 0 7px rgba(255,199,44,0.6);}
.abtn{font-size:16px;cursor:pointer;background:none;border:none;padding:2px;}
#ov{position:absolute;inset:0;z-index:50;background:rgba(3,8,18,0.9);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;cursor:pointer;border-radius:16px;}
.pc{width:70px;height:70px;background:linear-gradient(135deg,#B31B1B,#7a1212);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:28px;box-shadow:0 0 40px rgba(179,27,27,0.5);transition:transform 0.2s;}
#ov:hover .pc{transform:scale(1.08);}
.ot{font-family:'Orbitron',monospace;font-size:15px;color:#FFC72C;font-weight:700;}
.os{font-family:'Space Mono',monospace;font-size:11px;color:#4a7ab5;letter-spacing:1px;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
@keyframes sIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
@keyframes pIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes typeIn{from{opacity:0}to{opacity:1}}
</style></head><body>
<div class="vw">
  <div id="ov" onclick="enableAndPlay()">
    <div class="pc">▶</div>
    <div class="ot">Click to Play with Audio Narration</div>
    <div class="os">🔊 ENTERPRISE AI DEMO · 55 SECONDS · BUILT BY PRADEEP</div>
  </div>
  <div class="vtb">
    <div class="tl-row"><div class="tl tl-r"></div><div class="tl tl-y"></div><div class="tl tl-g"></div></div>
    <div class="vtitle">ENTERPRISE AI PLATFORM v3.1 — LIVE DEMO · PRADEEP</div>
    <div class="rec"><div class="rdot"></div> REC</div>
  </div>
  <div class="stage" id="stage">
    <!-- Scene 0: AI generates stories -->
    <div class="scene active" id="s0">
      <div class="term">
        <div class="thdr">▸ AI JIRA ASSISTANT — NATURAL LANGUAGE INPUT</div>
        <div class="tln"><span class="pr">PO@platform:~$</span> <span class="cm" id="tt"></span><span class="cur"></span></div>
        <div class="tln" id="r1" style="opacity:0;margin-top:10px"><span class="op">✦ Analysing requirement... Generating stories...</span></div>
        <div class="tln" id="r2" style="opacity:0;margin-top:4px"><span class="hl">✓ Created 5 user stories · 2 tasks · 1 epic · Sprint 14 assigned</span></div>
      </div>
    </div>
    <!-- Scene 1: Jira board with AI stories -->
    <div class="scene" id="s1">
      <div class="jboard">
        <div class="jhdr">
          <div class="jlogo">J</div>
          <div class="jpn">SPRINT 14 — AI GENERATED</div>
          <div class="aib">✦ AI AUTO-CREATED</div>
        </div>
        <div class="jcols">
          <div><div class="chdr">📋 To Do</div>
            <div class="tkt new"><div class="tid">PRJ-142</div><div class="ttl">OAuth2 login with MFA support</div><div class="tmeta"><span class="tp ph">High</span><div class="tav">SA</div></div></div>
            <div class="tkt new" style="animation-delay:.15s"><div class="tid">PRJ-143</div><div class="ttl">Mobile checkout Apple Pay</div><div class="tmeta"><span class="tp pm">Med</span><div class="tav">MK</div></div></div>
          </div>
          <div><div class="chdr">⚙️ In Progress</div>
            <div class="tkt"><div class="tid">PRJ-138</div><div class="ttl">API rate limiting middleware</div><div class="tmeta"><span class="tp ph">High</span><div class="tav">JP</div></div></div>
          </div>
          <div><div class="chdr">✅ Done</div>
            <div class="tkt"><div class="tid">PRJ-131</div><div class="ttl">Database schema migration v3</div><div class="tmeta"><span class="tp pd">Done</span><div class="tav">RL</div></div></div>
            <div class="tkt"><div class="tid">PRJ-129</div><div class="ttl">CI/CD pipeline setup</div><div class="tmeta"><span class="tp pd">Done</span><div class="tav">SA</div></div></div>
          </div>
        </div>
      </div>
    </div>
    <!-- Scene 2: PO Edit mode -->
    <div class="scene" id="s2">
      <div class="edit-scene">
        <div class="edit-hdr">
          <div style="font-size:16px;">✏️</div>
          <div class="edit-hdr-title">PO Edit Mode — Review & Refine AI Stories</div>
          <div class="edit-badge">EDITABLE</div>
        </div>
        <div class="edit-card">
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:#B31B1B;margin-bottom:8px;">PRJ-142 · Editing Story</div>
          <div class="edit-field active" id="ef1">OAuth2 login with MFA and biometric support</div>
          <div class="edit-field" id="ef2">As a user, I want to log in with MFA so that my account is secure</div>
          <div class="edit-row">
            <div class="edit-select">Priority: Critical ▼</div>
            <div class="edit-select">Points: 8 ▼</div>
            <div class="edit-select">Sprint 1 ▼</div>
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:#4a7ab5;margin-bottom:5px;">ACCEPTANCE CRITERIA</div>
          <div class="edit-field" style="font-size:9px;line-height:1.6;">• Given valid credentials, login succeeds within 2s<br>• Given MFA prompt, user can use TOTP or biometric<br>• Given invalid token, error is shown clearly</div>
          <button class="save-btn" id="saveBtn">💾 Save Changes</button>
        </div>
      </div>
    </div>
    <!-- Scene 3: Push to Jira -->
    <div class="scene" id="s3">
      <div class="push-scene">
        <div class="push-panel">
          <div class="push-title">🚀 Push to Jira Cloud</div>
          <div class="push-fields">
            <div class="push-field filled" id="pf1">https://myorg.atlassian.net</div>
            <div class="push-field filled" id="pf2">po@myorg.com</div>
            <div class="push-field filled" id="pf3">••••••••••••••• (API Token)</div>
            <div class="push-field filled" id="pf4">Project Key: PROJ</div>
          </div>
          <div class="push-progress">
            <div style="font-family:'Space Mono',monospace;font-size:9px;color:#FFC72C;margin-bottom:4px;">PUSHING STORIES <span id="pushCount">0</span>/5</div>
            <div class="push-bar-bg"><div class="push-bar-fill" id="pushBar" style="width:0%"></div></div>
          </div>
        </div>
        <div class="push-items" id="pushItems">
          <div class="push-item pending" id="pi0"><div class="pi-dot" style="background:#FFC72C;"></div><div class="pi-id">PRJ-142</div><div class="pi-title">OAuth2 login with MFA...</div><div class="pi-key" id="pk0">pending</div></div>
          <div class="push-item pending" id="pi1"><div class="pi-dot" style="background:#444;"></div><div class="pi-id">PRJ-143</div><div class="pi-title">Mobile checkout Apple Pay</div><div class="pi-key" id="pk1">—</div></div>
          <div class="push-item pending" id="pi2"><div class="pi-dot" style="background:#444;"></div><div class="pi-id">PRJ-144</div><div class="pi-title">Dashboard with recent orders</div><div class="pi-key" id="pk2">—</div></div>
          <div class="push-item pending" id="pi3"><div class="pi-dot" style="background:#444;"></div><div class="pi-id">PRJ-145</div><div class="pi-title">Email verification flow</div><div class="pi-key" id="pk3">—</div></div>
          <div class="push-item pending" id="pi4"><div class="pi-dot" style="background:#444;"></div><div class="pi-id">PRJ-146</div><div class="pi-title">Account profile management</div><div class="pi-key" id="pk4">—</div></div>
        </div>
      </div>
    </div>
    <!-- Scene 4: Impact dashboard -->
    <div class="scene" id="s4">
      <div class="mw">
        <div style="text-align:center;margin-bottom:16px;">
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:#B31B1B;letter-spacing:3px;text-transform:uppercase;margin-bottom:5px">Live Impact Dashboard</div>
          <div style="font-family:'Orbitron',monospace;font-size:16px;color:#e8f4ff;font-weight:700">This Sprint · Real-Time Results</div>
        </div>
        <div class="mgrid">
          <div class="mc"><div class="mv" id="m1">0</div><div class="ml">Tickets Pushed</div><div class="md">↑ 340% faster</div></div>
          <div class="mc"><div class="mv" id="m2">0</div><div class="ml">Records Processed</div><div class="md">↑ 98.6% accuracy</div></div>
          <div class="mc"><div class="mv" id="m3">0h</div><div class="ml">Hours Saved</div><div class="md">Per sprint</div></div>
        </div>
        <div class="ins">
          <div class="idot"></div>
          <div class="ilbl">AI INSIGHT:</div>
          <div class="itxt">Sprint 14 is <span style="color:#FFC72C">23% ahead of velocity</span>. 5 stories pushed to Jira in <span style="color:#69F0AE">8 seconds</span>. PO edited 2 stories before push — AI auto-applied Fibonacci points &amp; Gherkin AC.</div>
        </div>
      </div>
    </div>
  </div>
  <div class="capbar">
    <div class="cico">🔊</div>
    <div class="ctxt"><span class="cspk">NARRATOR</span><span id="ct">Welcome to the Enterprise AI Platform by Pradeep. Click Play to begin the narrated demo...</span></div>
  </div>
  <div class="ctrls">
    <button class="cb" onclick="prev()">⏮</button>
    <button class="pbtn" id="pb" onclick="togglePlay()">▶</button>
    <button class="cb" onclick="next()">⏭</button>
    <div class="pgbar" id="pg" onclick="seek(event)"><div class="pgfill" id="pf" style="width:0%"></div></div>
    <div class="tdisp" id="td">0:00 / 0:55</div>
    <button class="abtn" id="ab" onclick="toggleAudio()" title="Toggle narration">🔊</button>
    <div class="sdots">
      <div class="sd active" id="d0" onclick="goTo(0)"></div>
      <div class="sd" id="d1" onclick="goTo(1)"></div>
      <div class="sd" id="d2" onclick="goTo(2)"></div>
      <div class="sd" id="d3" onclick="goTo(3)"></div>
      <div class="sd" id="d4" onclick="goTo(4)"></div>
    </div>
  </div>
</div>
<script>
const narrs=[
  "Welcome to the Enterprise AI Platform built by Pradeep. Watch as a Product Owner types a plain English requirement — and AI instantly generates a complete Jira sprint with user stories, priorities, story points, and sprint assignments.",
  "In just seconds, the AI created 5 fully written user stories with Gherkin acceptance criteria, story points estimated, and Sprint 14 assigned — saving over 30 minutes of manual Product Owner work.",
  "Brand new in version 3.1 — PO Edit Mode. The Product Owner can now review every AI-generated story and edit the title, user story, priority, points, acceptance criteria, and subtasks before pushing to Jira — ensuring full PO control.",
  "Now watch the one-click Jira Push. Enter your Jira URL, email, and API token, and the platform pushes all reviewed stories — including PO edits — directly into your Jira project in seconds with full ADF formatting.",
  "Here is this sprint's live impact dashboard. 5 tickets pushed to Jira in 8 seconds, 2.4 million records processed by the ETL engine, and 18 engineering hours saved — all powered by AI so your team can focus on what truly matters."
];
const caps=[
  "A Product Owner types a plain-English feature request — AI instantly creates complete Jira tickets with priorities, story points, and sprint assignments.",
  "Done in seconds! AI generated 5 user stories with Gherkin AC, set priorities, estimated Fibonacci points, and assigned Sprint 14 — saving 30+ minutes.",
  "PO Edit Mode (NEW v3.1) — Edit title, user story, priority, points, sprint, acceptance criteria and subtasks before pushing. Full PO control over AI output.",
  "One-click Jira Push — Enter Jira URL + email + API token + project key, and all reviewed stories are pushed directly into Jira Cloud with proper ADF formatting.",
  "Live impact: 5 tickets pushed to Jira in 8 seconds · 2.4M records processed · 18 engineering hours saved — automatically, every sprint."
];
const dur=[11000,9000,12000,12000,11000];
const total=dur.reduce((a,b)=>a+b,0);
let cur=0,playing=false,elapsed=0,ticker=null,speechOn=true;
const sy=window.speechSynthesis;
function speak(i){if(!speechOn||!sy)return;sy.cancel();const u=new SpeechSynthesisUtterance(narrs[i]);u.rate=0.92;u.pitch=1.05;u.volume=1.0;const vs=sy.getVoices();const v=vs.find(v=>v.lang.startsWith('en')&&(v.name.includes('Daniel')||v.name.includes('Google US')||v.name.includes('Samantha')||v.name.includes('Alex')))||vs.find(v=>v.lang.startsWith('en'))||vs[0];if(v)u.voice=v;sy.speak(u);}
function stop(){sy&&sy.cancel();}
function toggleAudio(){speechOn=!speechOn;document.getElementById('ab').textContent=speechOn?'🔊':'🔇';if(speechOn&&playing)speak(cur);else stop();}
function updateCap(i){const el=document.getElementById('ct');el.style.opacity=0;setTimeout(()=>{el.textContent=caps[i];el.style.opacity=1;},280);}
const ts="Generate user stories for: mobile checkout with Apple Pay, email verification, and account dashboard with recent orders history.";
function animS0(){const e=document.getElementById('tt'),r1=document.getElementById('r1'),r2=document.getElementById('r2');r1.style.opacity=0;r2.style.opacity=0;e.textContent='';let i=0;const iv=setInterval(()=>{if(i<ts.length){e.textContent+=ts[i++];}else{clearInterval(iv);setTimeout(()=>r1.style.opacity=1,400);setTimeout(()=>r2.style.opacity=1,1200);}},36);}
function animS2(){
  // Simulate typing in edit field
  const ef1=document.getElementById('ef1');
  const origText='OAuth2 login with MFA support';
  const newText='OAuth2 login with MFA and biometric support';
  ef1.textContent=origText;
  setTimeout(()=>{
    let i=origText.length;
    const extra=' and biometric support';
    let j=0;
    ef1.style.borderColor='#FFC72C';
    const iv=setInterval(()=>{
      if(j<extra.length){ef1.textContent=origText+extra.substring(0,j+1);j++;}
      else{clearInterval(iv);}
    },60);
  },1200);
  // Simulate save button click highlight
  setTimeout(()=>{
    const btn=document.getElementById('saveBtn');
    if(btn){btn.style.background='linear-gradient(135deg,#FFC72C,#FFB300)';btn.style.color='#1a1a1a';}
    setTimeout(()=>{if(btn){btn.style.background='linear-gradient(135deg,#2E7D32,#1B5E20)';btn.style.color='white';}},1200);
  },5000);
}
function animS3(){
  const keys=['PROJ-201','PROJ-202','PROJ-203','PROJ-204','PROJ-205'];
  let done=0;
  keys.forEach((key,i)=>{
    setTimeout(()=>{
      const pi=document.getElementById('pi'+i);
      const pk=document.getElementById('pk'+i);
      const dot=pi?pi.querySelector('.pi-dot'):null;
      if(pi)pi.classList.add('ok');
      if(pk)pk.textContent=key;
      if(dot){dot.style.background='#69F0AE';}
      done++;
      document.getElementById('pushCount').textContent=done;
      document.getElementById('pushBar').style.width=(done/5*100)+'%';
    },i*1600+400);
  });
}
function animS4(){let v1=0,v2=0,v3=0;const go=()=>{v1=Math.min(5,v1+1);v2=Math.min(24,v2+1);v3=Math.min(18,v3+1);document.getElementById('m1').textContent=v1;document.getElementById('m2').textContent=(v2*100000).toLocaleString();document.getElementById('m3').textContent=v3+'h';if(v2<24||v3<18)requestAnimationFrame(go);};go();}
function showScene(i){document.querySelectorAll('.scene').forEach((s,j)=>s.classList.toggle('active',j===i));document.querySelectorAll('.sd').forEach((d,j)=>d.classList.toggle('active',j===i));updateCap(i);if(playing)speak(i);if(i===0)animS0();if(i===2)animS2();if(i===3)animS3();if(i===4)animS4();}
function goTo(i){cur=i;showScene(i);let p=0;for(let j=0;j<i;j++)p+=dur[j];elapsed=p;updD();}
function updD(){const pct=(elapsed/total)*100;document.getElementById('pf').style.width=pct+'%';const s=Math.floor(elapsed/1000),ts2=Math.floor(total/1000);document.getElementById('td').textContent=`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')} / ${Math.floor(ts2/60)}:${String(ts2%60).padStart(2,'0')}`;}
function startPlay(){playing=true;document.getElementById('pb').textContent='⏸';speak(cur);let p=0;for(let i=0;i<cur;i++)p+=dur[i];let sc=elapsed-p;ticker=setInterval(()=>{elapsed+=100;sc+=100;updD();if(sc>=dur[cur]){sc=0;cur++;if(cur>=dur.length){cur=0;elapsed=0;pausePlay();return;}showScene(cur);}},100);}
function pausePlay(){playing=false;clearInterval(ticker);stop();document.getElementById('pb').textContent='▶';}
function togglePlay(){if(playing)pausePlay();else startPlay();}
function prev(){pausePlay();cur=Math.max(0,cur-1);goTo(cur);}
function next(){pausePlay();cur=Math.min(4,cur+1);goTo(cur);}
function seek(e){const b=document.getElementById('pg'),r=b.getBoundingClientRect();const p=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));elapsed=Math.floor(p*total);let a=0;for(let i=0;i<dur.length;i++){if(elapsed<a+dur[i]){cur=i;break;}a+=dur[i];}showScene(cur);updD();}
function enableAndPlay(){document.getElementById('ov').style.display='none';const w=new SpeechSynthesisUtterance('');w.volume=0;sy.speak(w);setTimeout(()=>{showScene(0);startPlay();},150);}
if(sy.onvoiceschanged!==undefined)sy.onvoiceschanged=()=>{};
showScene(0);
</script></body></html>""", height=560, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">✦ Who Benefits From This Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:16px;">
      <div style="background:white;border:1px solid #E8E8E8;border-top:4px solid #B31B1B;border-radius:12px;padding:22px;box-shadow:0 4px 14px rgba(0,0,0,0.07);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
          <div style="width:46px;height:46px;background:linear-gradient(135deg,#B31B1B,#7a1212);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">🎯</div>
          <div><div style="font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:700;color:#B31B1B;">Product Owners</div><div style="font-size:10px;color:#999;letter-spacing:1px;font-family:'Space Mono',monospace;">AI JIRA ASSISTANT</div></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:9px;">
          <div style="background:#FFF8F8;border:1px solid #FFE0E0;border-left:3px solid #B31B1B;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">✍️ Auto-Generate User Stories</div><div style="font-size:12px;color:#555;line-height:1.5;">Type a feature in plain English — AI writes complete stories with Gherkin AC, story points, and sprint assignments.</div></div>
          <div style="background:#FFF8F8;border:1px solid #FFE0E0;border-left:3px solid #B31B1B;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">✏️ PO Edit Mode (NEW v3.1)</div><div style="font-size:12px;color:#555;line-height:1.5;">Review and edit every AI story — title, user story, priority, points, sprint, AC, and subtasks — before pushing to Jira.</div></div>
          <div style="background:#FFF8F8;border:1px solid #FFE0E0;border-left:3px solid #B31B1B;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">🚀 One-Click Jira Push (NEW v3.1)</div><div style="font-size:12px;color:#555;line-height:1.5;">Enter Jira URL, email, API token, and project key — push all stories directly into Jira Cloud in seconds with full ADF formatting.</div></div>
          <div style="background:#FFF8F8;border:1px solid #FFE0E0;border-left:3px solid #B31B1B;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">📊 One-Click Stakeholder Reports</div><div style="font-size:12px;color:#555;line-height:1.5;">Sprint summaries, release notes, and burndown narratives formatted for executives — zero manual writing required.</div></div>
        </div>
      </div>
      <div style="background:white;border:1px solid #E8E8E8;border-top:4px solid #1E90FF;border-radius:12px;padding:22px;box-shadow:0 4px 14px rgba(0,0,0,0.07);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
          <div style="width:46px;height:46px;background:linear-gradient(135deg,#1E4A80,#0d2137);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">⚙️</div>
          <div><div style="font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:700;color:#1565C0;">Data Engineers</div><div style="font-size:10px;color:#999;letter-spacing:1px;font-family:'Space Mono',monospace;">AI ETL BUILDER</div></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:9px;">
          <div style="background:#F0F7FF;border:1px solid #BBDEFB;border-left:3px solid #1E90FF;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">🔮 Auto-Pipeline Generation</div><div style="font-size:12px;color:#555;line-height:1.5;">Describe source and target in English — AI generates the full ETL pipeline code, connectors, and transformation logic.</div></div>
          <div style="background:#F0F7FF;border:1px solid #BBDEFB;border-left:3px solid #1E90FF;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">🔍 Schema Auto-Detection</div><div style="font-size:12px;color:#555;line-height:1.5;">Intelligently infers data types, detects anomalies, and maps source-to-target schemas — no manual configuration needed.</div></div>
          <div style="background:#F0F7FF;border:1px solid #BBDEFB;border-left:3px solid #1E90FF;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">🛡️ Smart Error Handling</div><div style="font-size:12px;color:#555;line-height:1.5;">AI monitors pipeline health, auto-retries failed jobs, and sends root-cause alerts before failures hit production.</div></div>
          <div style="background:#F0F7FF;border:1px solid #BBDEFB;border-left:3px solid #1E90FF;border-radius:8px;padding:11px 13px;"><div style="font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px;">📋 Auto-Documentation & Lineage</div><div style="font-size:12px;color:#555;line-height:1.5;">Lineage maps and transformation docs auto-update every pipeline run — always accurate, zero effort.</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Measurable ROI From Day One</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px;">
      <div style="background:white;border:1px solid #E8E8E8;border-top:3px solid #B31B1B;border-radius:10px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:#B31B1B;line-height:1;">~80%</div>
        <div style="font-size:10px;color:#999;letter-spacing:1px;text-transform:uppercase;margin:5px 0 3px;font-family:'Space Mono',monospace;">Time Saved on Tickets</div>
        <div style="font-size:11px;color:#555;line-height:1.4;">POs create Jira stories 5× faster with AI vs manual entry</div>
      </div>
      <div style="background:white;border:1px solid #E8E8E8;border-top:3px solid #2E7D32;border-radius:10px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:#2E7D32;line-height:1;">12×</div>
        <div style="font-size:10px;color:#999;letter-spacing:1px;text-transform:uppercase;margin:5px 0 3px;font-family:'Space Mono',monospace;">Pipeline Build Speed</div>
        <div style="font-size:11px;color:#555;line-height:1.4;">Build ETL pipelines in minutes, not half-days</div>
      </div>
      <div style="background:white;border:1px solid #E8E8E8;border-top:3px solid #1565C0;border-radius:10px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:#1565C0;line-height:1;">↓60%</div>
        <div style="font-size:10px;color:#999;letter-spacing:1px;text-transform:uppercase;margin:5px 0 3px;font-family:'Space Mono',monospace;">Sprint Planning Time</div>
        <div style="font-size:11px;color:#555;line-height:1.4;">AI handles grooming, estimation, and assignment automatically</div>
      </div>
      <div style="background:white;border:1px solid #E8E8E8;border-top:3px solid #E65100;border-radius:10px;padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-family:'Rajdhani',sans-serif;font-size:38px;font-weight:700;color:#E65100;line-height:1;">8s</div>
        <div style="font-size:10px;color:#999;letter-spacing:1px;text-transform:uppercase;margin:5px 0 3px;font-family:'Space Mono',monospace;">To Push 5 Stories</div>
        <div style="font-size:11px;color:#555;line-height:1.4;">One-click push to Jira Cloud with ADF formatting</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 What\'s Covered in the Demo (v3.1)</div>', unsafe_allow_html=True)
    for step, icon, title, desc in [
        ("01","⚡","Real-time ETL Execution Flow","Watch the GDE-style pipeline animate: Reading → Transforming → Done with live record counts."),
        ("02","📂","Multi-file Upload & Auto JOIN","Upload multiple CSVs — engine auto-detects joins and generates optimised pandas code via LLaMA 3.3 70B."),
        ("03","📋","AI Jira Breakdown by Project Type","Select project type, configure team, get full Epic + Stories + Gherkin AC + Subtasks in seconds."),
        ("04","✏️","PO Edit Mode (NEW)","Expand any story to edit title, user story, priority, points, sprint, acceptance criteria and subtasks. Save or reset to AI version."),
        ("05","🚀","One-Click Jira Push (NEW)","Enter Jira URL, email, API token and project key — test connection then push all stories directly to Jira Cloud with ADF formatting."),
        ("06","💾","Export in 5 Formats","ETL as CSV/Excel (with audit log). Jira as Excel (multi-sheet with push log), TXT, or JSON — always reflects PO edits."),
    ]:
        st.markdown(f"""<div style="background:white;border:1px solid #E8E8E8;border-left:4px solid #B31B1B;border-radius:8px;padding:14px 18px;margin:7px 0;box-shadow:0 2px 8px rgba(0,0,0,0.06);"><div style="display:flex;align-items:center;gap:12px;"><div style="background:#B31B1B;color:white;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;min-width:32px;">{step}</div><div style="font-size:20px;">{icon}</div><div><div style="font-size:13px;font-weight:700;color:#1a1a1a;">{title}</div><div style="font-size:12px;color:#555;margin-top:2px;line-height:1.5;">{desc}</div></div></div></div>""", unsafe_allow_html=True)


# ============================================================
# HISTORY
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">ETL Transformation History</div>', unsafe_allow_html=True)
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No transformations executed yet.")
