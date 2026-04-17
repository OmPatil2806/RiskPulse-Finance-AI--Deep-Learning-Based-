"""
RiskPulse Finance AI — Streamlit Web Application
Production-grade UI for mental health financial stress prediction.
"""

import os
import sys
import json
import pickle
import re
import numpy as np
import streamlit as st
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
ROOT_DIR   = APP_DIR.parent
SRC_DIR    = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
sys.path.insert(0, str(SRC_DIR))

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="RiskPulse Finance AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────────────────────
from risk_score import full_risk_report, RISK_BANDS

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {background: #0d1117;}
    [data-testid="stSidebar"]          {background: #161b22;}
    h1, h2, h3, h4                     {color: #e6edf3 !important;}
    p, li, label                       {color: #8b949e;}

    /* ── Cards ── */
    .risk-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
    }
    .metric-card {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    /* ── Risk gauge ── */
    .gauge-bar {
        height: 16px;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ecc71 0%, #f39c12 50%, #e74c3c 100%);
        margin: 8px 0;
    }
    .gauge-needle {
        height: 20px;
        margin-top: -18px;
        border-left: 3px solid white;
        border-radius: 2px;
    }

    /* ── Badges ── */
    .badge-low    {background:#1a472a; color:#2ecc71; padding:4px 12px; border-radius:20px; font-weight:700;}
    .badge-medium {background:#451a00; color:#f39c12; padding:4px 12px; border-radius:20px; font-weight:700;}
    .badge-high   {background:#450a0a; color:#e74c3c; padding:4px 12px; border-radius:20px; font-weight:700;}

    /* ── Input ── */
    textarea {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {background: linear-gradient(135deg, #2ea043, #3fb950);}
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    """Load BiLSTM model, tokenizer, and label encoder once."""
    import tensorflow as tf
    from feature_engineering import TextFeatureEngineer

    model = tf.keras.models.load_model(str(MODELS_DIR / "trained_model.keras"))
    fe    = TextFeatureEngineer.load(str(MODELS_DIR))

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODELS_DIR / "classes.json") as f:
        class_names = json.load(f)

    return model, fe, le, class_names


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(text: str, model, fe, le, class_names: list) -> dict:
    """Run inference and return full risk report."""
    cleaned = clean_text(text)
    seq     = fe.texts_to_sequences([cleaned])
    proba   = model.predict(seq, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_cls = class_names[pred_idx]
    return full_risk_report(pred_cls, proba, class_names)


def badge_html(level: str) -> str:
    cls = level.lower().replace(" ", "-").replace("risk", "").strip(" -")
    return f'<span class="badge-{cls}">{level}</span>'


def render_gauge(score: float, color: str) -> None:
    pct = score / 100
    st.markdown(f"""
    <div style="margin:12px 0;">
        <div class="gauge-bar"></div>
        <div style="width:{pct*100:.1f}%; border-left:3px solid {color}; height:20px; margin-top:-20px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;color:#8b949e;font-size:12px;">
        <span>0 — Low</span><span>50 — Medium</span><span>100 — High</span>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=72)
    st.markdown("## 💹 RiskPulse Finance AI")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    RiskPulse Finance AI analyses text for mental health indicators
    and converts them into actionable **Financial Stress Risk Scores**.

    **Risk Score Bands:**
    """)
    for cls, (lo, hi) in RISK_BANDS.items():
        st.markdown(f"- **{cls}**: {lo}–{hi}")
    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown("""
    🟢 **Low Risk** (0–35): Stable  
    🟡 **Medium Risk** (35–65): Monitor  
    🔴 **High Risk** (65–100): Act Now
    """)
    st.markdown("---")
    st.caption("v1.0.0 · RiskPulse Finance AI · © 2024")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 💹 RiskPulse Finance AI")
st.markdown("#### Financial Stress Risk Assessment powered by BiLSTM Deep Learning")
st.markdown("---")

# Load models
try:
    model, fe, le, class_names = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model not found. Please run `python src/train.py` first.\n\n{e}")
    model_loaded = False

# ── Input section ──────────────────────────────────────────────────────────────
col_input, col_examples = st.columns([3, 1])

with col_input:
    st.markdown("### 📝 Enter Patient / Client Statement")
    user_text = st.text_area(
        label="Statement",
        placeholder="Enter a statement here — e.g. 'I have been feeling overwhelmed by my finances and can't sleep…'",
        height=160,
        label_visibility="collapsed",
    )

with col_examples:
    st.markdown("### 💡 Examples")
    examples = {
        "😊 Normal":     "I feel great today! Work is going well and I'm looking forward to the weekend.",
        "😰 Anxiety":    "I can't stop worrying about my debt. Every bill fills me with dread and panic.",
        "😔 Depression": "Nothing feels worth it anymore. I've stopped caring about my finances entirely.",
        "🤔 Other":      "My thoughts race and I feel disconnected from reality sometimes.",
    }
    for label, text in examples.items():
        if st.button(label, key=f"ex_{label}"):
            st.session_state["example_text"] = text

if "example_text" in st.session_state:
    user_text = st.session_state.pop("example_text")

# ── Predict button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Analyse Financial Stress Risk", disabled=not model_loaded)

if predict_clicked:
    if not user_text or len(user_text.strip()) < 5:
        st.warning("⚠️ Please enter at least a few words.")
    else:
        with st.spinner("Analysing statement with BiLSTM model…"):
            result = predict(user_text.strip(), model, fe, le, class_names)

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # ── Row 1: Key metrics ────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:28px;">🏷️</div>
                <div style="color:#8b949e;font-size:12px;margin:4px 0;">Predicted Class</div>
                <div style="color:#e6edf3;font-size:20px;font-weight:700;">{result['predicted_class']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:28px;">📈</div>
                <div style="color:#8b949e;font-size:12px;margin:4px 0;">Risk Score</div>
                <div style="color:{result['risk_color']};font-size:28px;font-weight:700;">{result['risk_score']:.1f}</div>
                <div style="color:#8b949e;font-size:11px;">/ 100</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:28px;">⚠️</div>
                <div style="color:#8b949e;font-size:12px;margin:4px 0;">Risk Level</div>
                <div style="font-size:16px;font-weight:700;margin-top:6px;">{badge_html(result['risk_level'])}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            top_prob = max(result['probabilities'].values())
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:28px;">🎯</div>
                <div style="color:#8b949e;font-size:12px;margin:4px 0;">Confidence</div>
                <div style="color:#e6edf3;font-size:24px;font-weight:700;">{top_prob:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Gauge + Probabilities ───────────────────────────────────────
        col_gauge, col_probs = st.columns([1, 1])

        with col_gauge:
            st.markdown(f"""<div class="risk-card">
                <h3 style="color:#e6edf3;margin:0 0 12px 0;">🎚️ Financial Stress Risk Gauge</h3>""",
                unsafe_allow_html=True)
            render_gauge(result["risk_score"], result["risk_color"])
            st.markdown(f"""
                <div style="margin-top:16px;padding:12px;background:#21262d;border-radius:8px;">
                    <div style="color:#8b949e;font-size:12px;">💼 Financial Advisory</div>
                    <div style="color:#e6edf3;margin-top:6px;">{result['advice']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_probs:
            st.markdown('<div class="risk-card"><h3 style="color:#e6edf3;margin:0 0 16px 0;">📊 Class Probabilities</h3>',
                        unsafe_allow_html=True)
            colors_map = {"Normal": "#2ecc71", "Anxiety": "#f39c12",
                          "Depression": "#e74c3c", "Other": "#9b59b6"}
            for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                col = colors_map.get(cls, "#58a6ff")
                is_pred = "★ " if cls == result["predicted_class"] else ""
                st.markdown(f"""
                <div style="margin:8px 0;">
                    <div style="display:flex;justify-content:space-between;color:#e6edf3;font-size:13px;margin-bottom:4px;">
                        <span>{is_pred}{cls}</span><span style="color:{col};font-weight:700;">{prob:.1f}%</span>
                    </div>
                    <div style="background:#30363d;border-radius:4px;height:8px;">
                        <div style="background:{col};width:{prob}%;height:8px;border-radius:4px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Disclaimer ─────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-top:16px;">
            <span style="color:#8b949e;font-size:12px;">
            ⚠️ <strong>Disclaimer:</strong> RiskPulse Finance AI is a decision-support tool only.
            Risk scores are derived from NLP analysis and should not replace professional medical,
            psychological, or financial advice. Always consult qualified practitioners for clinical decisions.
            </span>
        </div>""", unsafe_allow_html=True)

elif not predict_clicked:
    # Welcome state
    st.markdown("""
    <div class="risk-card" style="text-align:center;padding:48px;">
        <div style="font-size:64px;margin-bottom:16px;">💹</div>
        <h2 style="color:#e6edf3;margin:0;">Welcome to RiskPulse Finance AI</h2>
        <p style="margin:16px 0;">Enter a client statement above and click <strong>Analyse</strong>
        to receive an AI-powered financial stress risk assessment.</p>
        <p style="color:#58a6ff;font-size:14px;">Powered by BiLSTM · TensorFlow/Keras · 4-Class Mental Health Classification</p>
    </div>
    """, unsafe_allow_html=True)
