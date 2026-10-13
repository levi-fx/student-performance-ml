import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------- Page Config ----------
st.set_page_config(
    page_title="Student Risk Predictor // ML-Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Load the saved models ----------
BASE = Path(__file__).parent


@st.cache_resource
def load_models():
    model = joblib.load(BASE / "app_model.pkl")
    scaler = joblib.load(BASE / "cluster_scaler.pkl")
    km = joblib.load(BASE / "kmeans.pkl")
    return model, scaler, km


app_model: GradientBoostingRegressor
scaler: StandardScaler
km: KMeans
app_model, scaler, km = load_models()

APP_COLS = ["studytime", "failures", "absences", "Dalc", "Walc", "goout", "G1", "G2"]
CLUSTER_COLS = APP_COLS + ["G3"]

# ---------- Name the clusters automatically ----------
centers = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_COLS
)
order = centers["G3"].sort_values().index.tolist()
names = {
    order[0]: "Academically struggling",
    order[1]: "Lifestyle risk",
    order[-1]: "On track",
}
advice = {
    "Academically struggling": "CRITICAL: Failure probability is elevated. Immediate academic counseling intervention and structured remediation required.",
    "Lifestyle risk": "Moderate outcome risk detected. High alcohol coefficients or attendance gaps are pulling the score down. Reallocating evening hours to coursework can reverse the regression decline.",
    "On track": "Strong core grades G1 & G2 insulate baseline mark. Student exhibits robust retention vectors and consistent attendance patterns.",
}

# =============================================
# INJECT STITCH DESIGN SYSTEM CSS
# =============================================
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

/* ── Design Tokens ── */
:root {
    --void:              #050505;
    --charcoal:          #111214;
    --graphite:          #1c1e21;
    --slate:             #2a2d31;
    --mid-grey:          #6b7078;
    --light-grey:        #b9bec6;
    --off-white:         #eceef1;
    --neon:              #39ff14;
    --neon-dim:          rgba(57,255,20,0.35);
    --neon-glow:         rgba(57,255,20,0.15);
    --alert-red:         #ff4d4d;
    --alert-red-dim:     rgba(255,77,77,0.35);
    --surface-glass:     rgba(28,30,33,0.55);
    --surface-glass-dim: rgba(17,18,20,0.65);
    --border-glass:      rgba(255,255,255,0.08);
    --border-glass-hi:   rgba(255,255,255,0.16);
    --warning-amber:     #ffd166;
}

/* ── Global Streamlit Overrides ── */
.stApp {
    background-color: var(--void) !important;
    background-image:
        radial-gradient(ellipse 60% 50% at 85% -10%, rgba(57,255,20,0.10), transparent 70%),
        radial-gradient(ellipse 50% 40% at -10% 110%, rgba(28,30,33,0.8), transparent 60%),
        radial-gradient(circle at center, transparent 40%, rgba(5,5,5,0.8) 100%);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif !important;
    color: var(--off-white) !important;
}

/* Tech grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-size: 40px 40px;
    background-image:
        linear-gradient(to right, rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,0.018) 1px, transparent 1px);
    pointer-events: none;
    z-index: 0;
    opacity: 0.7;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 1400px !important; padding-top: 0 !important; }
div[data-testid="stSidebar"] { display: none !important; }

/* ── Monospace Accent Font Utility ── */
.mono { font-family: 'JetBrains Mono', monospace !important; }
.mono-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}
.mono-metric {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: -0.02em;
}

/* ── Custom Nav Bar ── */
.nav-bar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 60px;
    background: rgba(18,19,21,0.6);
    backdrop-filter: blur(24px) saturate(140%);
    border-bottom: 1px solid var(--border-glass);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
    margin: -1rem -1rem 1.5rem -1rem;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--off-white);
}
.nav-brand .beacon {
    width: 8px;
    height: 8px;
    background: var(--neon);
    border-radius: 3px;
    box-shadow: 0 0 8px rgba(57,255,20,0.8);
}
.nav-brand .version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--mid-grey);
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: lowercase;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    background: rgba(57,255,20,0.08);
    border: 1px solid rgba(57,255,20,0.3);
    box-shadow: 0 0 14px rgba(57,255,20,0.12);
}
.nav-status .dot {
    width: 6px;
    height: 6px;
    background: var(--neon);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--neon);
    animation: pulse-beacon 2s ease-in-out infinite;
}
.nav-status span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--neon);
    font-weight: 500;
}

@keyframes pulse-beacon {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ── Glass Card ── */
.glass-card,
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(24px) saturate(140%) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 1.5rem !important;
    padding: 1.75rem !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6), 0 0 24px rgba(57,255,20,0.04) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: none !important;
    background: transparent !important;
}
.glass-card::before,
div[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    inset: 0;
    top: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), rgba(57,255,20,0.15), transparent);
    pointer-events: none;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}
.section-header h2 {
    font-family: 'Inter', sans-serif;
    font-size: 1.3rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--off-white);
    margin: 0;
}
.section-header .tag {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--mid-grey);
    background: var(--slate);
    padding: 0.25rem 0.6rem;
    border-radius: 0.35rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ── Slider Overrides ── */
div[data-testid="stSlider"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: rgba(236,238,241,0.9) !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--neon) !important;
    font-weight: 600 !important;
    background: rgba(57,255,20,0.1) !important;
    border: 1px solid rgba(57,255,20,0.25) !important;
    border-radius: 9999px !important;
    padding: 0.1rem 0.5rem !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background: var(--off-white) !important;
    border: 2px solid var(--neon) !important;
    box-shadow: 0 0 12px rgba(57,255,20,0.6) !important;
    width: 18px !important;
    height: 18px !important;
}
div[data-testid="stSlider"] div[data-testid="stTickBar"] > div {
    background: var(--graphite) !important;
}

/* ── Metric Overrides ── */
div[data-testid="stMetric"] {
    background: var(--surface-glass-dim);
    border: 1px solid var(--border-glass);
    border-radius: 1rem;
    padding: 1rem 1.25rem;
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--mid-grey) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    letter-spacing: -0.02em !important;
}

/* ── Button Overrides ── */
div.stButton > button {
    width: 100%;
    background: var(--neon) !important;
    color: var(--void) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.85rem 1.5rem !important;
    box-shadow: 0 0 24px rgba(57,255,20,0.45) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
div.stButton > button:hover {
    box-shadow: 0 0 35px rgba(57,255,20,0.7) !important;
    filter: brightness(1.1) !important;
    transform: scale(1.02);
}
div.stButton > button:active {
    transform: scale(0.98);
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.risk-badge .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse-beacon 2s ease-in-out infinite;
}
.risk-on-track {
    background: var(--neon);
    color: var(--void);
    box-shadow: 0 0 20px rgba(57,255,20,0.4);
}
.risk-on-track .pulse-dot { background: var(--void); }

.risk-lifestyle {
    background: transparent;
    border: 2px solid var(--neon);
    color: var(--neon);
    box-shadow: 0 0 15px rgba(57,255,20,0.2);
}
.risk-lifestyle .pulse-dot { background: var(--neon); }

.risk-struggling {
    background: var(--alert-red);
    color: var(--void);
    box-shadow: 0 0 20px rgba(255,77,77,0.4);
}
.risk-struggling .pulse-dot { background: var(--void); animation-name: pulse-beacon-fast; }

@keyframes pulse-beacon-fast {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(1.6); }
}

/* ── Habit Chips ── */
.habit-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    background: var(--graphite);
    border: 1px solid rgba(255,255,255,0.05);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.02em;
    color: var(--light-grey);
}
.habit-chip.warning {
    border-color: rgba(57,255,20,0.3);
    color: var(--neon);
}
.habit-chip.critical {
    border-color: rgba(255,77,77,0.3);
    color: var(--alert-red);
}

/* ── Advice Box ── */
.advice-box {
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    background: rgba(41,42,44,0.4);
    border-left: 3px solid var(--neon);
    border-top: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--off-white);
    line-height: 1.6;
}
.advice-box.critical {
    border-left-color: var(--alert-red);
}
.advice-box.lifestyle {
    border-left-color: var(--warning-amber);
}

/* ── Classification Tiers ── */
.tier-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
}
.tier-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 0.6rem;
    border-radius: 0.5rem;
    background: rgba(41,42,44,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    opacity: 0.5;
    transition: all 0.3s ease;
}
.tier-item.active {
    opacity: 1;
    border-color: rgba(57,255,20,0.4);
    background: rgba(57,255,20,0.08);
    box-shadow: 0 0 12px rgba(57,255,20,0.08);
}
.tier-item.active-red {
    opacity: 1;
    border-color: rgba(255,77,77,0.4);
    background: rgba(255,77,77,0.08);
}
.tier-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-bottom: 0.3rem;
}
.tier-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 600;
}
.tier-range {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    color: var(--mid-grey);
    margin-top: 0.15rem;
}

/* ── Confidence Bar ── */
.confidence-bar-bg {
    width: 100%;
    height: 6px;
    background: var(--graphite);
    border-radius: 9999px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    background: var(--neon);
    border-radius: 9999px;
    box-shadow: 0 0 8px var(--neon);
    transition: width 0.5s ease;
}

/* ── Score Display ── */
.score-display {
    text-align: center;
    padding: 1.5rem 1rem;
    background: rgba(13,14,16,0.4);
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
}
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    transition: all 0.3s ease;
}
.score-suffix {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    color: var(--mid-grey);
    font-weight: 500;
    margin-left: 0.25rem;
}
.score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--mid-grey);
    margin-bottom: 0.5rem;
}

/* Neon score color for on-track */
.score-neon { color: var(--neon); text-shadow: 0 0 25px rgba(57,255,20,0.45); }
.score-white { color: var(--off-white); }
.score-red { color: var(--alert-red); text-shadow: 0 0 25px rgba(255,77,77,0.45); }

/* ── Hero Section ── */
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 9999px;
    background: rgba(17,18,20,0.8);
    border: 1px solid rgba(57,255,20,0.35);
    box-shadow: 0 0 14px rgba(57,255,20,0.12);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.hero-pill .dot {
    width: 5px;
    height: 5px;
    background: var(--neon);
    border-radius: 50%;
}
.hero-pill span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--neon);
    font-weight: 500;
}

/* ── Footer ── */
.custom-footer {
    margin-top: 2rem;
    padding: 1rem 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.custom-footer span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--mid-grey);
}
.custom-footer .status {
    color: var(--neon);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.custom-footer .status .dot {
    width: 5px;
    height: 5px;
    background: var(--neon);
    border-radius: 50%;
}

/* ── Ambient Background Elements ── */
.ambient-ring {
    position: fixed;
    border-radius: 50%;
    border: 1px solid rgba(57,255,20,0.05);
    pointer-events: none;
    z-index: 0;
}

/* ── Coordinate overlays ── */
.coord-overlay {
    position: fixed;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(60,75,53,0.4);
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit default markdown paragraph margin */
.stMarkdown p { margin-bottom: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================
# AMBIENT BACKGROUND ELEMENTS
# =============================================
st.markdown(
    """
<div class="ambient-ring" style="top:80px;right:20%;width:450px;height:450px;"></div>
<div class="ambient-ring" style="top:75px;right:19.5%;width:465px;height:465px;border-style:dashed;"></div>
<div class="ambient-ring" style="bottom:-200px;left:-100px;width:500px;height:500px;border-color:rgba(60,75,53,0.1);"></div>
<div class="coord-overlay" style="top:85px;left:20px;">SYS_LOC // LAT.40.7128_N</div>
<div class="coord-overlay" style="bottom:80px;right:20px;">TENSOR STREAM SYNCHRONIZED // EPOCH T-2025</div>
""",
    unsafe_allow_html=True,
)

# =============================================
# NAVIGATION BAR
# =============================================
st.markdown(
    """
<div class="nav-bar">
    <div class="nav-brand">
        <div class="beacon"></div>
        <span>Student Risk Predictor</span>
        <span class="version">// ML-Engine v2.4</span>
    </div>
    <div class="nav-status">
        <div class="dot"></div>
        <span>Model Inference Active</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================
# SESSION STATE
# =============================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =============================================
# PAGE ROUTING
# =============================================
page = st.radio(
    "Select View",
    ["🏠 Overview", "🧬 Predictor"],
    key="nav_radio",
    horizontal=True,
    label_visibility="collapsed",
)

# ── Remove radio button default styling ──
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {
    gap: 0 !important;
}
div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.4rem 1rem !important;
    border-radius: 9999px !important;
    border: 1px solid var(--border-glass) !important;
    margin-right: 0.5rem !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label[data-checked="true"],
div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label:has(input:checked) {
    border-color: rgba(57,255,20,0.5) !important;
    background: rgba(57,255,20,0.08) !important;
    color: var(--neon) !important;
}
div[data-testid="stHorizontalBlock"] div[role="radiogroup"] label span[data-testid="stMarkdownContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================
# PAGE: OVERVIEW (LANDING)
# =============================================
if page == "🏠 Overview":
    st.markdown("")
    st.markdown("")

    # Hero pill
    st.markdown(
        """
    <div style="text-align:center;">
        <div class="hero-pill">
            <div class="dot"></div>
            <span>Machine Learning · Student Analytics</span>
            <span style="color:var(--mid-grey);font-size:0.5rem;">| INF-24</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Hero headline
    st.markdown(
        """
    <div style="text-align:center; margin-bottom:1rem;">
        <h1 style="font-family:'Inter',sans-serif; font-weight:600; font-size:3rem; letter-spacing:-0.04em; color:var(--off-white); line-height:1.15; margin:0;">
            Your marks, but<br>
            <span style="background: linear-gradient(90deg, var(--off-white), var(--neon), var(--off-white)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; filter: drop-shadow(0 0 35px rgba(57,255,20,0.25));">make it predictable.</span>
        </h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Subtitle
    st.markdown(
        """
    <div style="text-align:center; margin-bottom:2rem;">
        <p style="font-family:'Inter',sans-serif; font-size:1rem; color:var(--light-grey); max-width:600px; margin:0 auto; line-height:1.65;">
            Ever wondered what your final grade would be if your habits could talk? Ours can. A little machine learning, a few sliders, and zero judgment <span style="opacity:0.6;">(okay, a little)</span>.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Glass card — what the model does
    st.markdown(
        """<div class="glass-card" style="max-width:700px; margin:0 auto 1.5rem auto;">
<div class="section-header">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:1.2rem;">psychology</span>
<h2>What this model does</h2>
<div class="tag">Inference Engine Active</div>
</div>
<div style="display:flex;flex-direction:column;gap:0.75rem;">
<div style="display:flex;align-items:flex-start;gap:0.75rem;padding:0.5rem;border-radius:0.75rem;">
<div style="width:24px;height:24px;border-radius:0.5rem;background:var(--slate);border:1px solid rgba(57,255,20,0.3);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;box-shadow:0 0 8px rgba(57,255,20,0.2);">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:14px;">finance</span>
</div>
<div style="flex:1;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
<span style="font-size:0.85rem;font-weight:600;color:var(--off-white);">Predicts the final mark out of 20</span>
<span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;padding:0.2rem 0.6rem;border-radius:9999px;background:rgba(57,255,20,0.1);border:1px solid rgba(57,255,20,0.3);color:var(--neon);">[G3 Score: 0-20]</span>
</div>
</div>
<div style="display:flex;align-items:flex-start;gap:0.75rem;padding:0.5rem;border-radius:0.75rem;">
<div style="width:24px;height:24px;border-radius:0.5rem;background:var(--slate);border:1px solid rgba(57,255,20,0.3);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;box-shadow:0 0 8px rgba(57,255,20,0.2);">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:14px;">groups</span>
</div>
<div>
<span style="font-size:0.85rem;font-weight:600;color:var(--off-white);display:block;">Sorts students into a risk group</span>
<div style="display:flex;gap:0.4rem;margin-top:0.35rem;flex-wrap:wrap;">
<span class="habit-chip" style="color:var(--neon);border-color:rgba(57,255,20,0.2);background:rgba(57,255,20,0.08);">On track</span>
<span class="habit-chip" style="color:var(--warning-amber);border-color:rgba(255,209,102,0.2);background:rgba(255,209,102,0.08);">Lifestyle risk</span>
<span class="habit-chip critical">Academically struggling</span>
</div>
</div>
</div>
<div style="display:flex;align-items:flex-start;gap:0.75rem;padding:0.5rem;border-radius:0.75rem;">
<div style="width:24px;height:24px;border-radius:0.5rem;background:var(--slate);border:1px solid rgba(57,255,20,0.3);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;box-shadow:0 0 8px rgba(57,255,20,0.2);">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:14px;">flag</span>
</div>
<div style="flex:1;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
<span style="font-size:0.85rem;font-weight:600;color:var(--off-white);">Flags habits worth a closer look</span>
<span style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;color:var(--mid-grey);display:flex;align-items:center;gap:0.3rem;letter-spacing:0.06em;text-transform:uppercase;">
<span style="width:4px;height:4px;border-radius:50%;background:var(--neon);"></span>
Habit Analysis
</span>
</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # CTA Button to jump to Predictor
    def switch_to_predictor():
        st.session_state.nav_radio = "🧬 Predictor"

    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c2:
        st.button(
            "✦  Launch Risk Predictor  →",
            key="cta_launch",
            on_click=switch_to_predictor,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Ambient metadata
    st.markdown(
        """
    <div style="text-align:center;">
        <div style="display:inline-flex;align-items:center;gap:1rem;font-family:'JetBrains Mono',monospace;font-size:0.5rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--mid-grey);font-weight:600;">
            <span>EST. LATENCY: &lt; 14MS</span>
            <span>•</span>
            <span>ZERO REGISTRATION REQUIRED</span>
            <span>•</span>
            <span>395 STUDENT PORTFOLIOS</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================
# PAGE: PREDICTOR
# =============================================
elif page == "🧬 Predictor":

    col_input, col_output = st.columns([7, 5], gap="large")

    # ── LEFT: Input Parameters ──
    with col_input:
        st.markdown(
            """
        <div class="section-header">
            <span class="material-symbols-outlined" style="color:var(--neon);font-size:1.1rem;">tune</span>
            <h2>Student Habit & Academic Parameters</h2>
            <div class="tag">Dataset: PORT-STUDENT-395</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                studytime = st.slider(
                    "⏱ Study Time (1=low, 4=high)", 1, 4, 2, key="study"
                )
                failures = st.slider("📉 Past Class Failures", 0, 3, 0, key="fail")
                absences = st.slider("📅 Absences", 0, 75, 4, key="abs")
                Dalc = st.slider(
                    "🍺 Weekday Alcohol (1=low, 5=high)", 1, 5, 1, key="dalc"
                )

            with c2:
                Walc = st.slider(
                    "🍻 Weekend Alcohol (1=low, 5=high)", 1, 5, 2, key="walc"
                )
                goout = st.slider(
                    "🚶 Going Out (1=low, 5=high)", 1, 5, 3, key="goout"
                )
                G1 = st.slider("📊 Period 1 Grade (G1)", 0, 20, 10, key="g1")
                G2 = st.slider("📊 Period 2 Grade (G2)", 0, 20, 10, key="g2")

        st.markdown("")
        predict_clicked = st.button("✦  Compute Predictive Vector  →")

    # ── RIGHT: Results Panel ──
    with col_output:
        st.markdown(
            """<div class="section-header">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:1.1rem;">analytics</span>
<h2>Inference Output</h2>
<div class="tag">Predicted G3 Outcome</div>
</div>""",
            unsafe_allow_html=True,
        )

        if predict_clicked:
            st.session_state.predicted = True

        if st.session_state.predicted:
            # Run prediction
            student = pd.DataFrame(
                [[studytime, failures, absences, Dalc, Walc, goout, G1, G2]],
                columns=APP_COLS,
            )
            g3_pred = float(app_model.predict(student)[0])
            g3_pred = max(0.0, min(20.0, g3_pred))

            student["G3"] = g3_pred
            scaled = scaler.transform(student[CLUSTER_COLS])
            group = names[int(km.predict(scaled)[0])]

            # Determine display styling
            if group == "On track":
                score_class = "score-neon"
                badge_class = "risk-on-track"
                advice_class = "advice-box"
                tier_active = [True, False, False]
            elif group == "Lifestyle risk":
                score_class = "score-white"
                badge_class = "risk-lifestyle"
                advice_class = "advice-box lifestyle"
                tier_active = [False, True, False]
            else:
                score_class = "score-red"
                badge_class = "risk-struggling"
                advice_class = "advice-box critical"
                tier_active = [False, False, True]

            # Classification tiers
            tier_classes = ["", "", ""]
            tier_labels_color = [
                "color:var(--neon)",
                "color:var(--off-white)",
                "color:var(--alert-red)",
            ]
            for i, active in enumerate(tier_active):
                if active and i < 2:
                    tier_classes[i] = "active"
                elif active:
                    tier_classes[i] = "active-red"

            # Habit flags
            flags_html = []
            if Dalc >= 3:
                flags_html.append('<span class="habit-chip warning">⚠ High Weekday Alcohol</span>')
            if Walc >= 4:
                flags_html.append('<span class="habit-chip warning">⚠ High Weekend Alcohol</span>')
            if goout >= 4:
                flags_html.append('<span class="habit-chip warning">⚠ Frequent Going Out</span>')
            if absences >= 8:
                flags_html.append('<span class="habit-chip critical">🔴 Many Absences</span>')
            if studytime == 1:
                flags_html.append('<span class="habit-chip critical">🔴 Low Study Time</span>')
            if failures >= 2:
                flags_html.append(f'<span class="habit-chip critical">🔴 Prior Failures: {failures}</span>')
            if absences <= 5:
                flags_html.append('<span class="habit-chip" style="border-color:rgba(57,255,20,0.15);"><span style="color:var(--neon);">✓</span> Attendance On Track</span>')
            if G1 >= 12 and G2 >= 12:
                flags_html.append('<span class="habit-chip" style="border-color:rgba(57,255,20,0.15);"><span style="color:var(--neon);">✓</span> G1/G2 Strong Momentum</span>')

            flags_section = ""
            if flags_html:
                flags_section = f"""<div style="margin-top:1rem;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--mid-grey);margin-bottom:0.5rem;">Cognitive Factor Influence</div>
<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
{''.join(flags_html)}
</div>
</div>"""

            output_card_html = f"""<div class="glass-card">
<div class="score-display">
<div class="score-label">Final Term Estimate (G3)</div>
<span class="score-value {score_class}">{g3_pred:.1f}</span>
<span class="score-suffix">/ 20</span>
<div style="margin-top:1rem;display:flex;justify-content:center;">
<div class="{badge_class} risk-badge">
<div class="pulse-dot"></div>
<span>{group}</span>
</div>
</div>
</div>
<div style="margin-top:1rem;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--mid-grey);margin-bottom:0.5rem;">Classification Matrix Tiers:</div>
<div class="tier-grid">
<div class="tier-item {tier_classes[0]}">
<div class="tier-dot" style="background:var(--neon);box-shadow:0 0 6px var(--neon);"></div>
<div class="tier-label" style="{tier_labels_color[0]}">On Track</div>
<div class="tier-range">14.0 — 20.0</div>
</div>
<div class="tier-item {tier_classes[1]}">
<div class="tier-dot" style="border:1px solid var(--neon);"></div>
<div class="tier-label" style="{tier_labels_color[1]}">Lifestyle Risk</div>
<div class="tier-range">10.0 — 13.9</div>
</div>
<div class="tier-item {tier_classes[2]}">
<div class="tier-dot" style="background:var(--alert-red);"></div>
<div class="tier-label" style="{tier_labels_color[2]}">Struggling</div>
<div class="tier-range">0.0 — 9.9</div>
</div>
</div>
</div>
<div style="margin-top:1rem;">
<div class="{advice_class}">
<span class="material-symbols-outlined" style="color:var(--neon);font-size:1rem;vertical-align:middle;margin-right:0.5rem;">lightbulb</span>
{advice[group]}
</div>
</div>
{flags_section}
</div>"""
            st.markdown(output_card_html, unsafe_allow_html=True)

        else:
            # Waiting state
            st.markdown(
                """
            <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
                <div style="margin-bottom:1rem;">
                    <span class="material-symbols-outlined" style="font-size:3rem;color:var(--mid-grey);opacity:0.4;">psychology</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;color:var(--mid-grey);">
                    Awaiting Input Parameters
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:var(--light-grey);margin-top:0.5rem;">
                    Adjust the sliders and hit <strong>Compute Predictive Vector</strong> to run inference.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# =============================================
# FOOTER
# =============================================
st.markdown(
    """
<div class="custom-footer">
    <span>© 2025 Student Risk Predictor // Neural Analytics Lab. All cognitive weights reserved.</span>
    <div>
        <span>Gradient Boosting Regressor + KMeans // Trained on 395 Student Portfolios</span>
    </div>
    <div class="status">
        <div class="dot"></div>
        <span>Status: Online</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)