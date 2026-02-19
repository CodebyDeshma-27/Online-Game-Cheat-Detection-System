"""
dashboard.py
Run with: streamlit run dashboard.py
This is your real-time demo for the conference!
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

st.set_page_config(page_title="AI Cheat Detection", page_icon="🎮", layout="wide")

# ── LOAD MODELS ────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf           = joblib.load('models/random_forest_model.pkl')
    iso          = joblib.load('models/isolation_forest_model.pkl')
    scaler       = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return rf, iso, scaler, feature_cols

ENCODE_MAP = {
    'Gender':          {'Male': 1, 'Female': 0, 'Other': 2},
    'Location':        {'Brazil': 0, 'Germany': 1, 'India': 2, 'UK': 3, 'USA': 4},
    'GameGenre':       {'Battle Royale': 0, 'FPS': 1, 'MOBA': 2, 'RPG': 3, 'Sports': 4},
    'EngagementLevel': {'High': 0, 'Low': 1, 'Medium': 2}
}

def run_detection(player, rf, iso, scaler, feature_cols):
    row = dict(player)
    for col, mapping in ENCODE_MAP.items():
        row[col] = mapping.get(str(row.get(col, '')), 0)
    X = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X)
    start = time.time()
    rf_pred  = rf.predict(X_scaled)[0]
    rf_proba = rf.predict_proba(X_scaled)[0][1]
    iso_pred = iso.predict(X_scaled)[0]
    iso_score= -iso.score_samples(X_scaled)[0]
    latency  = (time.time() - start) * 1000
    is_cheat = (rf_pred == 0) or (iso_pred == -1)
    return is_cheat, rf_proba, iso_score, latency

# ── HEADER ─────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;color:#1565C0;'>🎮 AI-Based Intrusion Detection Framework</h1>
<h4 style='text-align:center;color:#666;'>Real-Time Multi-Genre Cheat Detection in Online Multiplayer Games</h4>
<h5 style='text-align:center;color:#999;'>Meenakshi Sundararajan Engineering College, Chennai</h5>
<hr>
""", unsafe_allow_html=True)

# Load models
models_ok = False
try:
    rf, iso, scaler, feature_cols = load_models()
    models_ok = True
except Exception as e:
    st.error(f"⚠️ Models not found! Run train_model.py first.\nError: {e}")

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Player", "📋 Batch CSV Scan", "📊 Model Results"])

# ══════════════════════════════════════════════════════════
# TAB 1 — SINGLE PLAYER DETECTION
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Player Stats for Real-Time Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Player Info**")
        player_id   = st.text_input("Player ID", "Player_001")
        age         = st.slider("Age", 13, 60, 22)
        gender      = st.selectbox("Gender", ["Male", "Female", "Other"])
        location    = st.selectbox("Location", ["USA", "India", "UK", "Germany", "Brazil"])
        genre       = st.selectbox("Game Genre", ["FPS", "Battle Royale", "MOBA", "RPG", "Sports"])
        engagement  = st.selectbox("Engagement Level", ["Low", "Medium", "High"])

    with col2:
        st.markdown("**Combat Stats**")
        kd          = st.slider("Kill/Death Ratio", 0.1, 20.0, 1.2, 0.1)
        headshot    = st.slider("Headshot %", 0, 100, 30)
        accuracy    = st.slider("Aim Accuracy %", 0, 100, 55)
        apm         = st.slider("Actions Per Minute", 10, 600, 80)

    with col3:
        st.markdown("**Movement & Network**")
        speed       = st.slider("Movement Speed", 1.0, 20.0, 5.5, 0.1)
        reaction    = st.slider("Reaction Time (ms)", 5, 600, 220)
        latency_net = st.slider("Network Latency (ms)", 5, 500, 60)
        packet_loss = st.slider("Packet Loss %", 0.0, 10.0, 0.5, 0.1)
        playtime    = st.slider("Playtime hrs/week", 1, 80, 25)
        session     = st.slider("Session Duration (min)", 5, 200, 45)
        achievement = st.slider("Achievement Rate %", 0, 100, 45)
        purchases   = st.checkbox("In-Game Purchases", True)

    if st.button("🔍 DETECT NOW", type="primary", use_container_width=True) and models_ok:
        player_data = {
            'Age': age, 'Gender': gender, 'Location': location,
            'GameGenre': genre, 'PlayTimeHours': playtime,
            'SessionDurationMinutes': session, 'KillDeathRatio': kd,
            'HeadshotPercentage': headshot, 'MovementSpeed': speed,
            'ReactionTimeMs': reaction, 'AimAccuracy': accuracy,
            'ActionsPerMinute': apm, 'NetworkLatencyMs': latency_net,
            'PacketLossPercentage': packet_loss,
            'InGamePurchases': int(purchases),
            'AchievementRate': achievement, 'EngagementLevel': engagement
        }

        is_cheat, rf_proba, iso_score, lat = run_detection(
            player_data, rf, iso, scaler, feature_cols)

        st.markdown("---")
        if is_cheat:
            st.markdown("<div style='background:#c62828;padding:18px;border-radius:10px;"
                        "text-align:center;'><h2 style='color:white;margin:0;'>"
                        "⛔ CHEATING DETECTED</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#2e7d32;padding:18px;border-radius:10px;"
                        "text-align:center;'><h2 style='color:white;margin:0;'>"
                        "✅ LEGITIMATE PLAYER</h2></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("RF Cheat Probability", f"{rf_proba:.1%}")
        c2.metric("Anomaly Score",        f"{iso_score:.4f}")
        c3.metric("Detection Latency",    f"{lat:.1f} ms")

        # Flags
        flags = []
        if kd > 5:        flags.append(f"🚩 KD Ratio = {kd} (too high)")
        if headshot > 70: flags.append(f"🚩 Headshot = {headshot}% (near perfect)")
        if reaction < 80: flags.append(f"🚩 Reaction = {reaction}ms (inhuman speed)")
        if accuracy > 90: flags.append(f"🚩 Aim = {accuracy}% (possible aimbot)")
        if apm > 250:     flags.append(f"🚩 APM = {apm} (macro/script detected)")
        if speed > 10:    flags.append(f"🚩 Speed = {speed} (speed hack)")

        if flags:
            st.warning("**Suspicious Indicators:**\n" + "\n".join(flags))

# ══════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV SCAN
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV to Scan Multiple Players")
    st.info("Upload gaming_dataset.csv or any CSV with the same columns.")

    uploaded = st.file_uploader("Choose CSV file", type=['csv'])

    if uploaded and models_ok:
        df_up = pd.read_csv(uploaded)
        st.write(f"**{len(df_up)} players loaded.**")

        results = []
        bar = st.progress(0)

        for i, (_, row) in enumerate(df_up.iterrows()):
            try:
                is_cheat, rf_p, iso_s, lat = run_detection(
                    row.to_dict(), rf, iso, scaler, feature_cols)
                results.append({
                    'PlayerID':   row.get('PlayerID', f'P{i}'),
                    'Verdict':    '⛔ CHEATING' if is_cheat else '✅ LEGITIMATE',
                    'RF_Prob':    f"{rf_p:.2%}",
                    'Anomaly':    f"{iso_s:.4f}",
                    'Latency_ms': f"{lat:.1f}"
                })
            except:
                results.append({'PlayerID': f'P{i}', 'Verdict': 'ERROR',
                                'RF_Prob': '-', 'Anomaly': '-', 'Latency_ms': '-'})
            bar.progress((i + 1) / len(df_up))

        df_res = pd.DataFrame(results)
        n_cheat = (df_res['Verdict'].str.contains('CHEAT')).sum()

        st.error(f"⛔ Cheaters detected: **{n_cheat}** out of {len(df_res)} players")
        st.dataframe(df_res, use_container_width=True)

        st.download_button("📥 Download Results",
                           df_res.to_csv(index=False),
                           "detection_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS & CHARTS
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance")

    try:
        res = joblib.load('models/results.pkl')
        rf_r   = res['random_forest']
        iso_r  = res['isolation_forest']
        comb_r = res['combined']

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Random Forest**")
            st.metric("Accuracy",  f"{rf_r['accuracy']:.2%}")
            st.metric("Precision", f"{rf_r['precision']:.2%}")
            st.metric("Recall",    f"{rf_r['recall']:.2%}")
            st.metric("F1-Score",  f"{rf_r['f1']:.2%}")
        with c2:
            st.markdown("**Isolation Forest**")
            st.metric("Accuracy",  f"{iso_r['accuracy']:.2%}")
            st.metric("Precision", f"{iso_r['precision']:.2%}")
            st.metric("Recall",    f"{iso_r['recall']:.2%}")
            st.metric("F1-Score",  f"{iso_r['f1']:.2%}")
        with c3:
            st.markdown("**Combined Framework**")
            st.metric("Accuracy",  f"{comb_r['accuracy']:.2%}")
            st.metric("Precision", f"{comb_r['precision']:.2%}")
            st.metric("Recall",    f"{comb_r['recall']:.2%}")
            st.metric("F1-Score",  f"{comb_r['f1']:.2%}")
            st.metric("False Positive Rate", f"{comb_r['fpr']:.2%}")
    except:
        st.warning("Run train_model.py first to see metrics.")

    # Show charts
    charts = [
        ('outputs/performance_comparison.png', 'Existing vs Proposed System'),
        ('outputs/confusion_matrix.png',       'Confusion Matrix'),
        ('outputs/roc_curve.png',              'ROC Curve'),
        ('outputs/feature_importance.png',     'Feature Importance'),
        ('outputs/behavior_distribution.png',  'Behavior Distribution'),
    ]
    for path, title in charts:
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_column_width=True)
