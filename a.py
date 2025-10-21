import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------- load model ----------
bundle = joblib.load("worklife_rf_smote_bundle.pkl")
model = bundle["model"]
feature_order = bundle["feature_order"]
# label_map = bundle.get("label_map", None)   # if you trained y as numbers

st.set_page_config(page_title="Work–Life Balance Predictor", page_icon="💼", layout="centered")
st.title("💼 Work–Life Balance Predictor")
st.caption("Predict balance level from your daily patterns")

# ---------- sidebar: instructions ----------
with st.sidebar:
    st.header("How to use")
    st.write("""
    1. Select your current levels for stress, social connections, and personal time satisfaction.  
    2. Pick your average **sleep hours**.  
    3. Click **Predict** to see your result & tips.
    """)
    st.markdown("---")
    st.write("Model: Random Forest + SMOTENC (balanced training)")

# ---------- inputs ----------
st.subheader("Enter your details")

ws = st.selectbox("Work Stress", ["Low", "Medium", "High"], index=1)
sc = st.selectbox("Social Connections", ["Weak", "Moderate", "Strong"], index=1)
pt = st.selectbox("Personal Time Satisfaction", ["Poor", "Average", "Good"], index=1)
slp = st.slider("Average Sleep Hours", min_value=4.5, max_value=9.0, value=7.0, step=0.1)

# mappings (must MATCH your training mapping)
ws_map = {'Low': 1, 'Medium': 2, 'High': 3}
sc_map = {'Weak': 1, 'Moderate': 2, 'Strong': 3}
pt_map = {'Poor': 1, 'Average': 2, 'Good': 3}

# build one-row dataframe in EXACT feature order
row = {
    "work stress": ws_map[ws],
    "social connections": sc_map[sc],
    "personal time satisfaction": pt_map[pt],
    "sleep_hours": float(slp),
}
# Align to training feature order & missing-safe
x_input = pd.DataFrame([row]).reindex(columns=feature_order, fill_value=0)

# ---------- predict ----------
if st.button("Predict"):
    pred = model.predict(x_input)[0]

    # If you trained y as numbers, map back to labels:
    # if label_map: pred = label_map[int(pred)]

    # probability (confidence)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_input)
        conf = float(np.max(proba) * 100.0)
    else:
        conf = None

    st.markdown("---")
    st.subheader("Result")

    if pred == "Balanced" or (isinstance(pred, str) and pred.lower().startswith("bal")):
        st.success("✅ Great! You have an excellent work–life balance.")
        st.write("""
        **Keep it up:**  
        • Maintain consistent routines and healthy sleep.  
        • Keep boundaries between work and personal time.  
        • Do a quick weekly check-in to keep balance steady.
        """)
    elif pred == "Moderate" or (isinstance(pred, str) and pred.lower().startswith("mod")):
        st.warning("⚖️ You’re managing okay, but there’s room to improve.")
        st.write("""
        **Try this:**  
        • Set clearer work cut-off time and take short breaks.  
        • Schedule one enjoyable activity daily (walk, music, hobby).  
        • Protect a small window for family/social time.
        """)
    else:
        st.error("❌ Poor work–life balance detected.")
        st.write("""
        **Action plan:**  
        • Aim for **7–8 hours** of sleep and reduce late-night screens.  
        • Add **10–20 min** of light exercise or meditation daily.  
        • Block **no-meeting / deep work** slots to reduce stress.  
        • Spend time with family/friends to decompress.
        """)

    if conf is not None:
        st.caption(f"Model confidence: **{conf:.1f}%**")

