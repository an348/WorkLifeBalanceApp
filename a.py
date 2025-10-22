import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------- load model safely ----------
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "worklife_rf_smote_bundle.pkl")

try:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_order = bundle["feature_order"]
    # label_map = bundle.get("label_map", None)   # if you trained y as numbers
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

st.set_page_config(page_title="Work–Life Balance Predictor", page_icon="💼", layout="centered")
st.title("💼 Work–Life Balance Predictor")
st.caption("Predict your work–life balance level based on daily patterns")

# ---------- sidebar instructions ----------
with st.sidebar:
    st.header("How to use")
    st.write("""
    1. Select your levels for stress, social connections, and personal time.  
    2. Choose your average **sleep hours**.  
    3. Click **Predict** to get your result & suggestions.
    """)
    st.markdown("---")
    st.write("Model: Random Forest + SMOTENC (balanced data)")

# ---------- input fields ----------
st.subheader("Enter your details")

ws = st.selectbox("Work Stress", ["Low", "Medium", "High"], index=1)
sc = st.selectbox("Social Connections", ["Weak", "Moderate", "Strong"], index=1)
pt = st.selectbox("Personal Time Satisfaction", ["Poor", "Average", "Good"], index=1)
slp = st.slider("Average Sleep Hours", min_value=4.5, max_value=9.0, value=7.0, step=0.1)

# ---------- mappings ----------
ws_map = {'Low': 3, 'Medium': 2, 'High': 1}
sc_map = {'Weak': 1, 'Moderate': 2, 'Strong': 3}
pt_map = {'Poor': 1, 'Average': 2, 'Good': 3}


# ---------- prepare input ----------
row = {
    "work stress": ws_map[ws],
    "social connections": sc_map[sc],
    "personal time satisfaction": pt_map[pt],
    "sleep_hours": float(slp)
}
x_input = pd.DataFrame([row]).reindex(columns=feature_order, fill_value=1)

# ---------- prediction ----------
# ---------- predict ----------
# ---------- predict ----------
if st.button("Predict"):
    try:
        # Make prediction
        pred = model.predict(x_input)[0]

        # --- Fix incorrect label encoding ---
        # Convert numeric or reversed labels into correct text
        if isinstance(pred, (int, float)):
            # Reverse mapping because model outputs may be inverted
            label_map = {0: "Balanced", 1: "Moderate", 2: "Poor"}
            pred = label_map.get(int(pred), "Unknown")
        else:
            # Reverse if string-based model output is flipped
            if str(pred).lower().startswith("poor"):
                pred = "Balanced"
            elif str(pred).lower().startswith("bal"):
                pred = "Poor"

        # Probability confidence
        if hasattr(model, "predict_proba"):
            conf = float(np.max(model.predict_proba(x_input)) * 100)
        else:
            conf = None

        st.markdown("---")
        st.subheader("Result")

        # Output message based on prediction
        if pred == "Balanced":
            st.success("✅ Great! You have an excellent work–life balance.")
            st.write("""
            **Keep it up:**  
            • Maintain consistent routines and healthy sleep.  
            • Keep boundaries between work and personal time.  
            • Do a quick weekly check-in to keep balance steady.
            """)
        elif pred == "Moderate":
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

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")


