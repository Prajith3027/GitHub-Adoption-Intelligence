import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – required for Streamlit
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import defaultdict

# ─────────────────────────────────────────
#  Page Config  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Adoption Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  Custom CSS – dark pro theme
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root palette (Premium Light Theme) ── */
:root {
    --bg:        #f8fafc;
    --surface:   #ffffff;
    --card:      #ffffff;
    --border:    #e2e8f0;
    --accent:    #3b82f6;
    --accent2:   #8b5cf6;
    --green:     #10b981;
    --red:       #ef4444;
    --yellow:    #f59e0b;
    --text:      #0f172a;
    --muted:     #64748b;
    --glow-blue: rgba(59,130,246,.06);
    --glow-purp: rgba(139,92,246,.04);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1300px !important; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 50%, #f1f5f9 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(139,92,246,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1d4ed8, #6d28d9, #1d4ed8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .6rem;
    position: relative; z-index: 1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 400;
    position: relative; z-index: 1;
    max-width: 620px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.2);
    color: #047857;
    font-size: .78rem;
    font-weight: 600;
    padding: .3rem .8rem;
    border-radius: 50px;
    margin-top: 1.2rem;
    position: relative; z-index: 1;
}

/* ── Search bar ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 1rem !important;
    padding: .75rem 1.2rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}
.stTextInput label { color: var(--muted) !important; font-weight: 500 !important; }

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: .7rem 2.2rem !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Section headings ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: .04em;
    text-transform: uppercase;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: .5rem;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    transition: transform .2s, box-shadow .2s, border-color .2s;
    box-shadow: 0 2px 8px rgba(0,0,0,.03);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,.06);
    border-color: var(--accent);
}
.metric-icon { font-size: 1.4rem; margin-bottom: .5rem; }
.metric-label {
    font-size: .75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .3rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    word-break: break-word;
}
.metric-value.accent  { color: #2563eb; }
.metric-value.green   { color: #059669; }
.metric-value.yellow  { color: #d97706; }
.metric-value.purple  { color: #7c3aed; }

/* ── Repo header card ── */
.repo-header {
    background: linear-gradient(135deg, var(--card), #f8fafc);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.02);
}
.repo-avatar {
    width: 56px; height: 56px;
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    flex-shrink: 0;
}
.repo-name { font-size: 1.35rem; font-weight: 700; color: var(--text); }
.repo-desc { font-size: .9rem; color: var(--muted); margin-top: .3rem; }
.repo-tag {
    display: inline-block;
    background: rgba(59,130,246,.06);
    border: 1px solid rgba(59,130,246,.15);
    color: #2563eb;
    font-size: .75rem;
    font-weight: 600;
    padding: .2rem .7rem;
    border-radius: 6px;
    margin: .4rem .3rem 0 0;
}

/* ── Result card ── */
.result-card {
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.result-card.high {
    background: linear-gradient(135deg, rgba(16,185,129,.06), rgba(16,185,129,.02));
    border: 1.5px solid rgba(16,185,129,.2);
    box-shadow: 0 10px 30px rgba(16,185,129,.04);
}
.result-card.low {
    background: linear-gradient(135deg, rgba(239,68,68,.06), rgba(239,68,68,.02));
    border: 1.5px solid rgba(239,68,68,.2);
    box-shadow: 0 10px 30px rgba(239,68,68,.04);
}
.result-emoji { font-size: 3.5rem; margin-bottom: .6rem; }
.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: .4rem;
}
.result-label.high { color: var(--green); }
.result-label.low  { color: var(--red); }
.result-sub { font-size: .95rem; color: var(--muted); }

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1.8rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: .85rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: .5rem;
}
.conf-val { color: var(--text); font-size: 1rem; }
.conf-track {
    background: #e2e8f0;
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}
.conf-fill.high { background: linear-gradient(90deg, #10b981, #34d399); }
.conf-fill.low  { background: linear-gradient(90deg, #ef4444, #f87171); }

/* ── Info pill ── */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(59,130,246,.06);
    border: 1px solid rgba(59,130,246,.15);
    color: #2563eb;
    font-size: .82rem;
    font-weight: 500;
    padding: .35rem .9rem;
    border-radius: 50px;
    margin: .3rem .25rem;
}

/* ── Warning / Error overrides ── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 6px; }

/* ── SHAP Explainability section ── */
.shap-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin-top: 1.8rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.03);
}
.shap-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: .55rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.shap-row:last-child { border-bottom: none; }
.shap-rank {
    font-size: .75rem;
    font-weight: 700;
    color: var(--muted);
    width: 22px;
    text-align: center;
    flex-shrink: 0;
}
.shap-name {
    font-size: .88rem;
    font-weight: 600;
    color: var(--text);
    width: 175px;
    flex-shrink: 0;
}
.shap-val {
    font-size: .78rem;
    color: var(--muted);
    width: 90px;
    flex-shrink: 0;
}
.shap-bar-wrap {
    flex: 1;
    background: #f1f5f9;
    border-radius: 50px;
    height: 9px;
    overflow: hidden;
}
.shap-bar-pos { height: 100%; border-radius: 50px; background: linear-gradient(90deg,#10b981,#34d399); }
.shap-bar-neg { height: 100%; border-radius: 50px; background: linear-gradient(90deg,#ef4444,#f87171); }
.shap-direction {
    font-size: .75rem;
    font-weight: 700;
    width: 95px;
    flex-shrink: 0;
    text-align: right;
}
.shap-direction.pos { color: #059669; }
.shap-direction.neg { color: #dc2626; }

/* ── Trend Analysis section ── */
.trend-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.trend-card.increasing {
    background: linear-gradient(135deg, rgba(16,185,129,.06), rgba(16,185,129,.02));
    border: 1.5px solid rgba(16,185,129,.2);
    box-shadow: 0 10px 30px rgba(16,185,129,.04);
}
.trend-card.stable {
    background: linear-gradient(135deg, rgba(59,130,246,.06), rgba(59,130,246,.02));
    border: 1.5px solid rgba(59,130,246,.2);
    box-shadow: 0 10px 30px rgba(59,130,246,.04);
}
.trend-card.decreasing {
    background: linear-gradient(135deg, rgba(239,68,68,.06), rgba(239,68,68,.02));
    border: 1.5px solid rgba(239,68,68,.2);
    box-shadow: 0 10px 30px rgba(239,68,68,.04);
}
.trend-emoji { font-size: 3.5rem; margin-bottom: .6rem; }
.trend-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: .4rem;
}
.trend-label.increasing { color: var(--green); }
.trend-label.stable { color: var(--accent); }
.trend-label.decreasing { color: var(--red); }
.trend-sub { font-size: .95rem; color: var(--muted); }

.trend-signal-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.trend-signal-row:last-child {
    border-bottom: none;
}
.trend-signal-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text);
}
.trend-signal-val {
    font-size: 0.9rem;
    font-weight: 700;
}
.trend-signal-val.pos { color: #059669; }
.trend-signal-val.neu { color: var(--muted); }
.trend-signal-val.neg { color: #dc2626; }

/* ── Final Insight section ── */
.insight-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-top: 2rem;
    color: #f1f5f9;
    box-shadow: 0 10px 30px rgba(15,23,42,.15);
}
.insight-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
}
.insight-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.insight-badge {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
}
.insight-badge.high-inc { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.insight-badge.high-dec { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
.insight-badge.low-inc { background: rgba(59,130,246,.15); color: #60a5fa; border: 1px solid rgba(59,130,246,.3); }
.insight-badge.low-dec { background: rgba(239,68,68,.15); color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.insight-text {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #cbd5e1;
}
</style>

""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load Models
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    return (
        joblib.load("models/lightgbm_model.pkl"),
        joblib.load("models/scaler.pkl"),
        joblib.load("models/language_encoder.pkl"),
        joblib.load("models/license_encoder.pkl"),
        joblib.load("models/adoption_encoder.pkl"),
    )

model, scaler, language_encoder, license_encoder, adoption_encoder = load_models()

# ─────────────────────────────────────────
#  Helper – safe label encoding
# ─────────────────────────────────────────
def safe_encode(encoder, value, fallback=None):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    if fallback and fallback in encoder.classes_:
        return encoder.transform([fallback])[0]
    # Check common fallbacks, or default to the first class
    for fb in ["Unknown", "No License", encoder.classes_[0]]:
        if fb in encoder.classes_:
            return encoder.transform([fb])[0]
    return 0

# ─────────────────────────────────────────
#  Helper – fetch trend data
# ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trend_data(owner, repo, headers):
    trend_data = {
        "commit_activity": None,
        "code_frequency": None,
        "releases": None,
        "recent_issues": None
    }
    
    # 1. Commit Activity (retry up to 3 times for 202 status code)
    commit_url = f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity"
    for attempt in range(3):
        try:
            r = requests.get(commit_url, headers=headers, timeout=10)
            if r.status_code == 200:
                trend_data["commit_activity"] = r.json()
                break
            elif r.status_code == 202:
                time.sleep(2)
            else:
                break
        except Exception:
            break
            
    # 2. Code Frequency (retry up to 3 times for 202 status code)
    code_freq_url = f"https://api.github.com/repos/{owner}/{repo}/stats/code_frequency"
    for attempt in range(3):
        try:
            r = requests.get(code_freq_url, headers=headers, timeout=10)
            if r.status_code == 200:
                trend_data["code_frequency"] = r.json()
                break
            elif r.status_code == 202:
                time.sleep(2)
            else:
                break
        except Exception:
            break
            
    # 3. Releases
    try:
        releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page=20"
        r = requests.get(releases_url, headers=headers, timeout=10)
        if r.status_code == 200:
            trend_data["releases"] = r.json()
    except Exception:
        pass
        
    # 4. Recent Issues
    try:
        issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page=50&sort=created&direction=desc"
        r = requests.get(issues_url, headers=headers, timeout=10)
        if r.status_code == 200:
            trend_data["recent_issues"] = r.json()
    except Exception:
        pass
        
    return trend_data

# ─────────────────────────────────────────
#  Hero Section
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🚀 GitHub Adoption Intelligence</div>
    <div class="hero-sub">
        Paste any GitHub repository URL and our LightGBM model will instantly
        predict its community adoption potential based on real repository signals.
    </div>
    <div class="badge">✦ Model Ready &nbsp;·&nbsp; LightGBM Classifier &nbsp;·&nbsp; 89.2 % Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Input Row
# ─────────────────────────────────────────
col_input, col_btn = st.columns([5, 1], gap="small")

with col_input:
    repo_url = st.text_input(
        "Repository URL",
        placeholder="https://github.com/owner/repository",
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    predict_button = st.button("Analyze ⚡", use_container_width=True)

# ─────────────────────────────────────────
#  Prediction Logic
# ─────────────────────────────────────────
if predict_button:
    if not repo_url.strip():
        st.warning("⚠️ Please enter a GitHub repository URL to continue.")
        st.stop()

    try:
        parts = repo_url.strip("/").split("/")
        if len(parts) < 5:
            st.error("❌ Invalid GitHub URL. Format: `https://github.com/owner/repo`")
            st.stop()

        owner, repo = parts[-2], parts[-1]

        with st.spinner("🔍 Fetching repository data…"):

            # ── Fetch repo metadata ──
            headers = {"Accept": "application/vnd.github+json"}
            
            # Use GitHub token from Streamlit secrets if available to prevent rate limits
            try:
                if "GITHUB_TOKEN" in st.secrets:
                    headers["Authorization"] = f"token {st.secrets['GITHUB_TOKEN']}"
                elif "github_token" in st.secrets:
                    headers["Authorization"] = f"token {st.secrets['github_token']}"
            except Exception:
                pass


            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}", headers=headers
            )
            if resp.status_code != 200:
                st.error("❌ Repository not found. Please check the URL.")
                st.stop()

            rd = resp.json()

            # ── Parse fields ──
            language         = rd.get("language") or "Unknown"
            forks            = rd.get("forks_count", 0)
            open_issues      = rd.get("open_issues_count", 0)
            size_kb          = rd.get("size", 0)
            stars            = rd.get("stargazers_count", 0)
            watchers         = rd.get("watchers_count", 0)
            description      = rd.get("description") or ""
            homepage         = rd.get("homepage") or ""
            topics           = rd.get("topics") or []
            license_name     = (rd["license"]["name"] if rd.get("license") else "No License")
            default_branch   = rd.get("default_branch", "main")
            is_fork          = rd.get("fork", False)

            created_at       = datetime.strptime(rd["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            pushed_at        = datetime.strptime(rd["pushed_at"],  "%Y-%m-%dT%H:%M:%SZ")
            project_age_days = (datetime.utcnow() - created_at).days
            days_since_push  = (datetime.utcnow() - pushed_at).days

            # ── Fetch contributors (paginated) ──
            contributors = 0
            page = 1
            while True:
                cr = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contributors"
                    f"?per_page=100&page={page}&anon=true",
                    headers=headers,
                )
                cd = cr.json()
                if not isinstance(cd, list) or len(cd) == 0:
                    break
                contributors += len(cd)
                page += 1

            # ── Fetch trend data ──
            trend_data = fetch_trend_data(owner, repo, headers)

        # ─────────────────────────────────────────
        #  Repo Header Card
        # ─────────────────────────────────────────
        tags_html = "".join(
            f'<span class="repo-tag">{t}</span>' for t in topics[:6]
        )
        lang_tag = f'<span class="repo-tag">⌨ {language}</span>'
        lic_tag  = f'<span class="repo-tag">📄 {license_name}</span>'

        st.markdown(f"""
        <div class="repo-header">
            <div class="repo-avatar">📦</div>
            <div style="flex:1; min-width:0;">
                <div class="repo-name">{owner} / {repo}</div>
                <div class="repo-desc">{description[:160] + ("…" if len(description) > 160 else "") if description else "No description provided."}</div>
                <div style="margin-top:.5rem">{lang_tag}{lic_tag}{tags_html}</div>
            </div>
            <div style="text-align:right; flex-shrink:0;">
                <div class="metric-label">⭐ Stars</div>
                <div style="font-size:1.8rem;font-weight:800;color:#f5a623;">{stars:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────
        #  Metrics Grid
        # ─────────────────────────────────────────
        st.markdown('<div class="section-heading">📊 Repository Signals</div>', unsafe_allow_html=True)

        def _card(icon, label, value, color=""):
            return f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
            </div>"""

        # Row 1
        r1 = (
            _card("🍴", "Forks",          f"{forks:,}",         "accent")
            + _card("🐛", "Open Issues",   f"{open_issues:,}",   "yellow")
            + _card("👥", "Contributors",  f"{contributors:,}",  "green")
            + _card("👁", "Watchers",      f"{watchers:,}",      "purple")
        )
        st.markdown(f'<div class="metric-grid">{r1}</div>', unsafe_allow_html=True)

        # Row 2
        created_date_str = created_at.strftime("%b %d, %Y")
        pushed_date_str  = pushed_at.strftime("%b %d, %Y")
        r2 = (
            _card("📅", "Created On",        created_date_str,              "")
            + _card("🚀", "Last Push",       pushed_date_str, "yellow" if days_since_push > 365 else "green")
            + _card("💾", "Repo Size",       f"{size_kb:,} KB",             "")
            + _card("🌿", "Default Branch",  default_branch,                "accent")
        )
        st.markdown(f'<div class="metric-grid">{r2}</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────
        #  Model Prediction
        # ─────────────────────────────────────────
        # Handle unseen language label
        if language not in language_encoder.classes_:
            language_used = "Unknown"
            st.info(
                f"ℹ️ Language **'{language}'** is not in the training vocabulary — "
                f"using **'Unknown'** as fallback for prediction."
            )
        else:
            language_used = language

        features = {
            "language":             safe_encode(language_encoder, language_used),
            "forks":                forks,
            "open_issues":          open_issues,
            "contributors":         contributors,
            "size_kb":              size_kb,
            "license":              safe_encode(license_encoder, license_name, fallback="No License"),
            "project_age_days":     project_age_days,
            "days_since_last_push": days_since_push,
            "description_length":   len(description),
            "created_year":         created_at.year,
            "pushed_year":          pushed_at.year,
        }

        df_input = pd.DataFrame([features])
        # Ensure exact column ordering as used in LightGBM model training
        df_input = df_input[[
            "language", "forks", "open_issues", "contributors", "size_kb", "license",
            "project_age_days", "days_since_last_push", "description_length", "created_year", "pushed_year"
        ]]

        # Predict using LightGBM model directly on raw features
        pred     = model.predict(df_input)
        proba    = model.predict_proba(df_input)[0]
        label    = adoption_encoder.inverse_transform(pred)[0]

        is_high    = (label == "High Adoption")
        css_class  = "high" if is_high else "low"
        emoji      = "🚀" if is_high else "📉"
        conf_pct   = float(proba.max() * 100)
        high_pct   = float(proba[0] * 100)   # index 0 = High Adoption (alphabetical)
        low_pct    = float(proba[1] * 100)

        # ─────────────────────────────────────────
        #  SHAP – Compute explanations
        # ─────────────────────────────────────────
        # TreeExplainer is the native, exact explainer for LightGBM.
        #
        # IMPORTANT – SHAP output format differs by version:
        #   Older SHAP  → list of 2 arrays: [class_0_shap, class_1_shap]
        #   Newer SHAP  → single ndarray of shape (n_samples, n_features)
        #                 representing SHAP values for class 1 (Low Adoption)
        #                 because class 0=High Adoption, class 1=Low Adoption (alphabetical)
        #
        # In both cases we normalise to "SHAP values toward the predicted class":
        #   - Predicted Low  Adoption (class 1): use values as-is   (positive → Low)
        #   - Predicted High Adoption (class 0): negate the values  (positive → High)
        shap_explainer      = shap.TreeExplainer(model)
        shap_values_all     = shap_explainer.shap_values(df_input)
        predicted_class_idx = int(pred[0])   # 0 = High Adoption, 1 = Low Adoption

        if isinstance(shap_values_all, list):
            # Old SHAP format: list of arrays, one per class
            # shap_values_all[k][row_index] → shape (11,)
            shap_vals_for_pred = np.array(shap_values_all[predicted_class_idx][0])
        else:
            # New SHAP format: single ndarray, shape (n_samples, n_features)
            # Values represent class 1 (Low Adoption)
            raw_sv = np.array(shap_values_all[0])   # shape (11,) for 1 sample
            if predicted_class_idx == 1:
                # Predicted Low Adoption → values already oriented correctly
                shap_vals_for_pred = raw_sv
            else:
                # Predicted High Adoption → negate so positive = pushes toward High
                shap_vals_for_pred = -raw_sv

        FEATURE_NAMES = [
            "language", "forks", "open_issues", "contributors", "size_kb",
            "license", "project_age_days", "days_since_last_push",
            "description_length", "created_year", "pushed_year"
        ]

        # Pair feature names with their SHAP values, sort by absolute magnitude
        shap_pairs = sorted(
            zip(FEATURE_NAMES, shap_vals_for_pred),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top5 = shap_pairs[:5]

        # ── friendly display names ──
        DISPLAY_NAMES = {
            "language":             "Language",
            "forks":                "Forks",
            "open_issues":          "Open Issues",
            "contributors":         "Contributors",
            "size_kb":              "Repo Size (KB)",
            "license":              "License",
            "project_age_days":     "Project Age (days)",
            "days_since_last_push": "Days Since Last Push",
            "description_length":   "Description Length",
            "created_year":         "Created Year",
            "pushed_year":          "Pushed Year",
        }

        # ── actual raw values for tooltip ──
        raw_vals = {
            "language":             language_used,
            "forks":                forks,
            "open_issues":          open_issues,
            "contributors":         contributors,
            "size_kb":              size_kb,
            "license":              license_name,
            "project_age_days":     project_age_days,
            "days_since_last_push": days_since_push,
            "description_length":   len(description),
            "created_year":         created_at.year,
            "pushed_year":          pushed_at.year,
        }

        # ─────────────────────────────────────────
        #  Adoption Trend Analysis Computation
        # ─────────────────────────────────────────
        trend_score = 0
        trend_signals = {}
        
        # Signal 1: Commit velocity (last 13 weeks vs previous 39 weeks)
        commit_activity = trend_data.get("commit_activity")
        if commit_activity and isinstance(commit_activity, list) and len(commit_activity) >= 52:
            totals = [week.get("total", 0) for week in commit_activity]
            recent_13 = totals[-13:]
            prior_39 = totals[:-13]
            
            avg_recent = sum(recent_13) / len(recent_13) if recent_13 else 0
            avg_prior = sum(prior_39) / len(prior_39) if prior_39 else 0
            
            if avg_prior > 0:
                pct_change = (avg_recent - avg_prior) / avg_prior
            else:
                pct_change = 0.0 if avg_recent == 0 else 1.0
                
            if pct_change >= 0.20:
                sig_score = 1
                sig_text = f"▲ Active (+{pct_change*100:.1f}%)"
                sig_class = "pos"
            elif pct_change <= -0.20:
                sig_score = -1
                sig_text = f"▼ Decreasing ({pct_change*100:.1f}%)"
                sig_class = "neg"
            else:
                sig_score = 0
                sig_text = f"Stable ({pct_change*100:+.1f}%)"
                sig_class = "neu"
                
            trend_score += sig_score
            trend_signals["commit_velocity"] = (sig_text, sig_class)
        else:
            trend_signals["commit_velocity"] = ("No active commit data", "neu")
            
        # Signal 2: Release cadence
        releases = trend_data.get("releases")
        if releases and isinstance(releases, list):
            release_dates = []
            for r in releases:
                pub_at = r.get("published_at")
                if pub_at:
                    try:
                        r_date = datetime.strptime(pub_at, "%Y-%m-%dT%H:%M:%SZ")
                        release_dates.append(r_date)
                    except Exception:
                        pass
            
            now = datetime.utcnow()
            recent_rels = sum(1 for d in release_dates if (now - d).days <= 180)
            prior_rels = sum(1 for d in release_dates if 180 < (now - d).days <= 360)
            
            if recent_rels >= 1:
                if recent_rels >= prior_rels:
                    sig_score = 1
                    sig_text = f"▲ Frequent ({recent_rels} recent)"
                    sig_class = "pos"
                else:
                    sig_score = 0
                    sig_text = f"Stable ({recent_rels} recent, was {prior_rels})"
                    sig_class = "neu"
            else:
                if prior_rels >= 2:
                    sig_score = -1
                    sig_text = "▼ Inactive (none recent)"
                    sig_class = "neg"
                else:
                    sig_score = 0
                    sig_text = "Stable (none recent)"
                    sig_class = "neu"
            trend_score += sig_score
            trend_signals["release_cadence"] = (sig_text, sig_class)
        else:
            trend_signals["release_cadence"] = ("No releases", "neu")
            
        # Signal 3: Issue/PR momentum (open issues created in last 90 days)
        recent_issues = trend_data.get("recent_issues")
        if recent_issues and isinstance(recent_issues, list):
            now = datetime.utcnow()
            created_dates = []
            for item in recent_issues:
                c_at = item.get("created_at")
                if c_at:
                    try:
                        date_c = datetime.strptime(c_at, "%Y-%m-%dT%H:%M:%SZ")
                        created_dates.append(date_c)
                    except Exception:
                        pass
            
            issues_last_90 = sum(1 for d in created_dates if (now - d).days <= 90)
            
            if issues_last_90 >= 10:
                sig_score = 1
                sig_text = f"▲ Active ({issues_last_90} opened)"
                sig_class = "pos"
            elif issues_last_90 == 0:
                sig_score = -1
                sig_text = "▼ Stagnant (0 opened)"
                sig_class = "neg"
            else:
                sig_score = 0
                sig_text = f"Moderate ({issues_last_90} opened)"
                sig_class = "neu"
            trend_score += sig_score
            trend_signals["issue_momentum"] = (sig_text, sig_class)
        else:
            trend_signals["issue_momentum"] = ("No issue activity", "neu")
            
        # Signal 4: Last Push activity
        if days_since_push <= 30:
            sig_score = 1
            sig_text = f"▲ Fresh ({days_since_push} days ago)"
            sig_class = "pos"
        elif days_since_push > 180:
            sig_score = -1
            sig_text = f"▼ Old ({days_since_push} days ago)"
            sig_class = "neg"
        else:
            sig_score = 0
            sig_text = f"Stable ({days_since_push} days ago)"
            sig_class = "neu"
        trend_score += sig_score
        trend_signals["push_recency"] = (sig_text, sig_class)
        
        # Determine Trend Status
        if trend_score >= 1:
            trend_status = "Increasing"
            trend_class = "increasing"
            trend_emoji = "📈"
        elif trend_score <= -2:
            trend_status = "Decreasing"
            trend_class = "decreasing"
            trend_emoji = "📉"
        else:
            trend_status = "Stable"
            trend_class = "stable"
            trend_emoji = "➡️"

        # ─────────────────────────────────────────
        #  Result Card
        # ─────────────────────────────────────────
        st.markdown('<div class="section-heading">🤖 Prediction Result</div>', unsafe_allow_html=True)

        result_col, info_col = st.columns([1.2, 1], gap="large")

        with result_col:
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label {css_class}">{label}</div>
                <div class="result-sub">
                    {"This repository shows strong community adoption signals." if is_high
                     else "This repository has limited community traction so far."}
                </div>
                <div class="conf-wrap">
                    <div class="conf-label">
                        <span>Model Confidence</span>
                        <span class="conf-val">{conf_pct:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill {css_class}" style="width:{conf_pct:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with info_col:
            st.markdown("""
            <div style="padding-top:.5rem">
            <div class="section-heading" style="font-size:.85rem;margin-top:.5rem">📈 Probability Breakdown</div>
            """, unsafe_allow_html=True)

            # High bar
            st.markdown(f"""
            <div style="margin-bottom:1rem">
                <div class="conf-label">
                    <span style="color:#10d98a;font-weight:700">🟢 High Adoption</span>
                    <span class="conf-val">{high_pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill high" style="width:{high_pct:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Low bar
            st.markdown(f"""
            <div style="margin-bottom:1.5rem">
                <div class="conf-label">
                    <span style="color:#f04361;font-weight:700">🔴 Low Adoption</span>
                    <span class="conf-val">{low_pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill low" style="width:{low_pct:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key signals pills
            st.markdown('<div class="section-heading" style="font-size:.85rem">🔑 Key Signals Used</div>', unsafe_allow_html=True)
            signals = [
                ("🍴", f"{forks:,} forks"),
                ("👥", f"{contributors:,} contributors"),
                ("📅", f"since {created_date_str}"),
                ("🚀", f"pushed {pushed_date_str}"),
                ("💾", f"{size_kb} KB"),
                ("🐛", f"{open_issues} issues"),
            ]
            pills = "".join(
                f'<span class="info-pill">{icon} {text}</span>'
                for icon, text in signals
            )
            st.markdown(f'<div>{pills}</div></div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────
        #  SHAP – Display: Top Factors (text)
        # ─────────────────────────────────────────
        st.markdown(
            '<div class="section-heading">🧠 Explainable AI · Why this prediction?</div>',
            unsafe_allow_html=True,
        )

        # Header label adapts to predicted class
        pred_class_label = "High Adoption" if predicted_class_idx == 0 else "Low Adoption"
        pred_class_color = "#059669" if predicted_class_idx == 0 else "#dc2626"

        st.markdown(f"""
        <div class="shap-section">
            <div style="margin-bottom:1.2rem;">
                <span style="font-size:.95rem;font-weight:700;color:{pred_class_color};">
                    Top factors driving the <em>{pred_class_label}</em> prediction:
                </span>
                <span style="font-size:.78rem;color:#94a3b8;margin-left:.6rem;">
                    (SHAP values — positive = pushes toward predicted class)
                </span>
            </div>
        """, unsafe_allow_html=True)

        # max abs SHAP for scaling bars to 100 %
        max_abs_shap = max(abs(v) for _, v in top5) if top5 else 1.0

        rows_html = ""
        for rank, (feat, sv) in enumerate(top5, start=1):
            bar_pct   = min(abs(sv) / max_abs_shap * 100, 100)
            bar_class = "shap-bar-pos" if sv >= 0 else "shap-bar-neg"
            dir_label = "▲ Positive impact" if sv >= 0 else "▼ Negative impact"
            dir_class = "pos" if sv >= 0 else "neg"
            raw_v     = raw_vals[feat]
            # Format raw value neatly
            if isinstance(raw_v, float):
                raw_str = f"{raw_v:,.2f}"
            elif isinstance(raw_v, int):
                raw_str = f"{raw_v:,}"
            else:
                raw_str = str(raw_v)

            rows_html += f"""
            <div class="shap-row">
                <span class="shap-rank">#{rank}</span>
                <span class="shap-name">{DISPLAY_NAMES[feat]}</span>
                <span class="shap-val">= {raw_str}</span>
                <div class="shap-bar-wrap">
                    <div class="{bar_class}" style="width:{bar_pct:.1f}%"></div>
                </div>
                <span class="shap-direction {dir_class}">{dir_label}</span>
            </div>"""

        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

        # ─────────────────────────────────────────
        #  SHAP – Display: Full Feature Bar Chart
        # ─────────────────────────────────────────
        st.markdown(
            '<div class="section-heading">📊 SHAP Feature Importance · All Features</div>',
            unsafe_allow_html=True,
        )

        # Sort all features by SHAP value (not abs) for diverging bar chart
        all_pairs_sorted = sorted(shap_pairs, key=lambda x: x[1])  # ascending
        feat_labels = [DISPLAY_NAMES[f] for f, _ in all_pairs_sorted]
        feat_shap   = [v for _, v in all_pairs_sorted]
        bar_colors  = ["#ef4444" if v < 0 else "#10b981" for v in feat_shap]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        bars = ax.barh(feat_labels, feat_shap, color=bar_colors,
                       height=0.55, edgecolor="none", zorder=3)

        # Value annotations
        for bar, val in zip(bars, feat_shap):
            pad   = 0.0005 * (max(feat_shap) - min(feat_shap) + 1e-9)
            xpos  = bar.get_width() + pad if val >= 0 else bar.get_width() - pad
            align = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center", ha=align,
                    fontsize=8, color="#475569", fontweight="600")

        ax.axvline(0, color="#cbd5e1", linewidth=1.2, zorder=4)
        ax.set_xlabel(
            f"SHAP value  (positive → {pred_class_label})",
            fontsize=9, color="#64748b", labelpad=8
        )
        ax.tick_params(axis="y", labelsize=9, colors="#0f172a")
        ax.tick_params(axis="x", labelsize=8, colors="#94a3b8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e2e8f0")
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(axis="x", color="#e2e8f0", linewidth=0.7, zorder=0)
        ax.set_title(
            f"SHAP Feature Impact — {pred_class_label} Prediction",
            fontsize=11, fontweight="700", color="#0f172a", pad=12
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)   # free memory

        # ─────────────────────────────────────────
        #  Adoption Trend Analysis Section
        # ─────────────────────────────────────────
        st.markdown(
            '<div class="section-heading">📈 Adoption Trend Analysis</div>',
            unsafe_allow_html=True,
        )
        
        trend_col_card, trend_col_signals = st.columns([1.2, 1], gap="large")
        
        with trend_col_card:
            st.markdown(f"""
            <div class="trend-card {trend_class}">
                <div class="trend-emoji">{trend_emoji}</div>
                <div class="trend-label {trend_class}">{trend_status}</div>
                <div class="trend-sub">
                    Based on historical commit activity, release cadence, and issue velocity over time.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with trend_col_signals:
            st.markdown(f"""
            <div style="padding-top:.5rem">
            <div class="section-heading" style="font-size:.85rem;margin-top:.5rem">🔑 Trend Indicators</div>
            
            <div class="trend-signal-row">
                <span class="trend-signal-name">Commit Velocity (Last 13 wks)</span>
                <span class="trend-signal-val {trend_signals['commit_velocity'][1]}">{trend_signals['commit_velocity'][0]}</span>
            </div>
            <div class="trend-signal-row">
                <span class="trend-signal-name">Release Cadence</span>
                <span class="trend-signal-val {trend_signals['release_cadence'][1]}">{trend_signals['release_cadence'][0]}</span>
            </div>
            <div class="trend-signal-row">
                <span class="trend-signal-name">Issue Momentum (Last 90 days)</span>
                <span class="trend-signal-val {trend_signals['issue_momentum'][1]}">{trend_signals['issue_momentum'][0]}</span>
            </div>
            <div class="trend-signal-row">
                <span class="trend-signal-name">Last Push Activity</span>
                <span class="trend-signal-val {trend_signals['push_recency'][1]}">{trend_signals['push_recency'][0]}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Render charts for Trend
        chart_col1, chart_col2 = st.columns(2, gap="medium")
        
        with chart_col1:
            if commit_activity and isinstance(commit_activity, list) and len(commit_activity) >= 52:
                totals = [week.get("total", 0) for week in commit_activity]
                weeks_label = [f"W{i+1}" for i in range(len(totals))]
                
                fig_c, ax_c = plt.subplots(figsize=(6, 3))
                fig_c.patch.set_facecolor("#ffffff")
                ax_c.set_facecolor("#f8fafc")
                
                ax_c.plot(weeks_label, totals, color="#3b82f6", linewidth=2, label="Weekly Commits")
                ax_c.axvspan(len(totals)-13, len(totals)-1, color="#10b981", alpha=0.15, label="Last 13 Weeks")
                
                ax_c.set_title("Weekly Commit Activity (Last 12 Months)", fontsize=10, fontweight="700")
                ax_c.spines["top"].set_visible(False)
                ax_c.spines["right"].set_visible(False)
                ax_c.spines["left"].set_color("#e2e8f0")
                ax_c.spines["bottom"].set_color("#e2e8f0")
                ax_c.grid(axis="y", color="#e2e8f0", linestyle="--", linewidth=0.7)
                
                quarter_ticks = [0, 13, 26, 39, 51]
                ax_c.set_xticks(quarter_ticks)
                ax_c.set_xticklabels(["1 yr ago", "9 mo ago", "6 mo ago", "3 mo ago", "Now"], fontsize=8)
                ax_c.tick_params(axis="both", labelsize=8)
                ax_c.legend(fontsize=8, loc="upper left")
                
                plt.tight_layout()
                st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)
            else:
                st.info("ℹ️ Weekly commit activity trend is not available or cached yet by GitHub.")
                
        with chart_col2:
            code_frequency = trend_data.get("code_frequency")
            if code_frequency and isinstance(code_frequency, list) and len(code_frequency) > 0:
                recent_cf = code_frequency[-52:]
                weeks = range(len(recent_cf))
                adds = [cf[1] for cf in recent_cf]
                dels = [abs(cf[2]) for cf in recent_cf]
                
                fig_cf, ax_cf = plt.subplots(figsize=(6, 3))
                fig_cf.patch.set_facecolor("#ffffff")
                ax_cf.set_facecolor("#f8fafc")
                
                ax_cf.fill_between(weeks, adds, color="#10b981", alpha=0.3, label="Code Insertions")
                ax_cf.plot(weeks, adds, color="#10b981", linewidth=1.5)
                ax_cf.fill_between(weeks, dels, color="#ef4444", alpha=0.3, label="Code Deletions")
                ax_cf.plot(weeks, dels, color="#ef4444", linewidth=1.5)
                
                ax_cf.set_title("Weekly Code Volume Additions vs Deletions", fontsize=10, fontweight="700")
                ax_cf.spines["top"].set_visible(False)
                ax_cf.spines["right"].set_visible(False)
                ax_cf.spines["left"].set_color("#e2e8f0")
                ax_cf.spines["bottom"].set_color("#e2e8f0")
                ax_cf.grid(axis="y", color="#e2e8f0", linestyle="--", linewidth=0.7)
                
                quarter_ticks_cf = [0, len(recent_cf)//4, len(recent_cf)//2, 3*len(recent_cf)//4, len(recent_cf)-1]
                ax_cf.set_xticks(quarter_ticks_cf[:len(recent_cf)])
                ax_cf.set_xticklabels(["Start", "Q2", "Q3", "Q4", "Now"][:len(recent_cf)], fontsize=8)
                ax_cf.tick_params(axis="both", labelsize=8)
                ax_cf.legend(fontsize=8, loc="upper left")
                
                plt.tight_layout()
                st.pyplot(fig_cf, use_container_width=True)
                plt.close(fig_cf)
            else:
                st.info("ℹ️ Code frequency volume trend is not available or cached yet by GitHub.")
                
        # Monthly Issues/PR Activity Chart (Full row width)
        if recent_issues and isinstance(recent_issues, list) and len(recent_issues) > 0:
            now = datetime.utcnow()
            issues_by_month = defaultdict(int)
            
            for item in recent_issues:
                c_at = item.get("created_at")
                if c_at:
                    try:
                        date_c = datetime.strptime(c_at, "%Y-%m-%dT%H:%M:%SZ")
                        months_ago = (now.year - date_c.year) * 12 + now.month - date_c.month
                        if 0 <= months_ago < 6:
                            month_str = date_c.strftime("%b %Y")
                            issues_by_month[(months_ago, month_str)] += 1
                    except Exception:
                        pass
                        
            if issues_by_month:
                sorted_months = sorted(issues_by_month.keys(), key=lambda x: x[0], reverse=True)
                labels = [m[1] for m in sorted_months]
                counts = [issues_by_month[m] for m in sorted_months]
                
                fig_i, ax_i = plt.subplots(figsize=(12, 3))
                fig_i.patch.set_facecolor("#ffffff")
                ax_i.set_facecolor("#f8fafc")
                
                bars_i = ax_i.bar(labels, counts, color="#8b5cf6", width=0.4, zorder=3)
                
                for bar in bars_i:
                    yval = bar.get_height()
                    ax_i.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", 
                             va='bottom', ha='center', fontsize=8, color="#475569", fontweight="600")
                             
                ax_i.set_title("Recent Issue & PR Activity (Last 6 Months)", fontsize=10, fontweight="700")
                ax_i.spines["top"].set_visible(False)
                ax_i.spines["right"].set_visible(False)
                ax_i.spines["left"].set_color("#e2e8f0")
                ax_i.spines["bottom"].set_color("#e2e8f0")
                ax_i.grid(axis="y", color="#e2e8f0", linestyle="--", linewidth=0.7, zorder=0)
                ax_i.tick_params(axis="both", labelsize=8)
                
                plt.tight_layout()
                st.pyplot(fig_i, use_container_width=True)
                plt.close(fig_i)
            else:
                st.info("ℹ️ Recent issue activity data is not sufficient to draw a monthly trend.")
        else:
            st.info("ℹ️ Recent issues/PRs timeline data is not available or could not be loaded.")
            
        # ─────────────────────────────────────────
        #  Final AI Insight Card Section
        # ─────────────────────────────────────────
        st.markdown(
            '<div class="section-heading">💡 Final AI Insight</div>',
            unsafe_allow_html=True,
        )
        
        if is_high:
            if trend_status == "Increasing":
                badge_class = "high-inc"
                badge_text = "High Adoption & Growing"
                insight_desc = f"The model predicts **High Adoption** ({conf_pct:.1f}% confidence) and recent activity confirms this upward trajectory. Stars, forks, and development activity are moving positively. This is a very active project."
            elif trend_status == "Decreasing":
                badge_class = "high-dec"
                badge_text = "High Adoption & Slowing"
                insight_desc = f"The model predicts **High Adoption** ({conf_pct:.1f}% confidence) based on its current signals, but monitor activity closely — recent developer updates, releases, or issue momentum show slowing pace. A deceleration might impact long-term support."
            else:
                badge_class = "high-inc"
                badge_text = "High Adoption & Stable"
                insight_desc = f"The model predicts **High Adoption** ({conf_pct:.1f}% confidence) with a **Stable** ongoing activity trend. The project maintains consistent updates and engagement, indicating a mature, well-supported repository."
        else:
            if trend_status == "Increasing":
                badge_class = "low-inc"
                badge_text = "Low Adoption & Rising"
                insight_desc = f"The model predicts **Low Adoption** ({conf_pct:.1f}% confidence) based on overall stats, but recent activity is trending upward — this repo shows potential future growth. If stars and commit velocity continue to climb, community adoption will likely follow."
            elif trend_status == "Decreasing":
                badge_class = "low-dec"
                badge_text = "Low Adoption & Declining"
                insight_desc = f"The model predicts **Low Adoption** ({conf_pct:.1f}% confidence) and recent activity confirms limited community traction. Pushes are infrequent, and there are few signs of active developer engagement or release updates."
            else:
                badge_class = "low-dec"
                badge_text = "Low Adoption & Stable"
                insight_desc = f"The model predicts **Low Adoption** ({conf_pct:.1f}% confidence) with stable, quiet ongoing activity. The repository remains in a maintenance or inactive state with a steady but low level of engagement."
                
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-header">
                <span class="insight-title">🤖 AI Integration Summary</span>
                <span class="insight-badge {badge_class}">{badge_text}</span>
            </div>
            <div class="insight-text">
                {insight_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div style="background:rgba(240,67,97,.1);border:1.5px solid rgba(240,67,97,.4);
                    border-radius:14px;padding:1.2rem 1.5rem;margin-top:1rem;">
            <div style="font-weight:700;color:#f04361;margin-bottom:.4rem;">⚠️ An Error Occurred</div>
            <div style="color:#64748b;font-size:.9rem;">{e}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid var(--border);color:#64748b;font-size:.8rem;">
    GitHub Adoption Intelligence &nbsp;·&nbsp; Powered by LightGBM &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)