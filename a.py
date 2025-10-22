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
if st.button("Predict"):
    try:
        pred = model.predict(x_input)[0]

        # Probability confidence
        if hasattr(model, "predict_proba"):
            conf = float(np.max(model.predict_proba(x_input)) * 100)
        else:
            conf = None

        st.markdown("---")
        st.subheader("Result")

        if pred == "Balanced" or (isinstance(pred, str) and pred.lower().startswith("bal")):
            st.success("✅ Great! You have an excellent work–life balance.")
            st.write("""
            **Keep it up:**  
            • Maintain consistent routines and good sleep.  
            • Keep clear boundaries between work and rest.  
            • Do weekly self-check-ins to stay balanced.
            """)
        elif pred == "Moderate" or (isinstance(pred, str) and pred.lower().startswith("mod")):
            st.warning("⚖️ You’re doing okay, but there’s room for improvement.")
            st.write("""
            **Try this:**  
            • Create a cut-off time for work.  
            • Schedule one enjoyable activity daily.  
            • Protect time for family and friends.
            """)
        else:
            st.error("❌ Poor work–life balance detected.")
            st.write("""
            **Action plan:**  
            • Sleep 7–8 hours nightly.  
            • Try short meditation or exercise.  
            • Reduce late-night screen use.  
            • Schedule small breaks during the day.
            """)

        if conf is not None:
            st.caption(f"Model confidence: **{conf:.1f}%**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
