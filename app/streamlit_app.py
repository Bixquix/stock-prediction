import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# 🔧 Model Configuration
# ==============================
MODEL_DIR = "../models/"
DATA_PATH = "../data/processed/web_feed.csv"
SCALER_PATH = "../models/feature_scaler.pkl"

MODELS = {
    "top1_AdaBoost.pkl": "AdaBoost ",
    "top6_LightGBM.pkl": "LightGBM ",
    "top5_ExtraTrees.pkl": "Extra Trees(57)",
    "top4_RandomForest.pkl": "Random Forest",
    "top8_XGBoost.pkl": "XGBoost ",
    "top7_SVM.pkl": "Support Vector Machine⭐ (Best Trader)",
    "top2_GradientBoost.pkl": "Gradient Boosting",
    "top3_LogisticRegression.pkl": "Logistic Regression(57)"
}

# ==============================
# 🎨 Page Setup
# ==============================
st.set_page_config(page_title="NIFTY50 Predictor", page_icon="📈", layout="wide")
st.title("📈 NIFTY50 Next-Day Direction Prediction App")
st.markdown(
    "This app uses **7 Machine Learning Models** to predict whether NIFTY50 will go "
    "**Up 📈** or **Down 📉** on the next trading day."
)

# ==============================
# 📥 Load Scaler
# ==============================
scaler = joblib.load(SCALER_PATH)

# ==============================
# 📥 Load All Models
# ==============================
loaded_models = {}
st.subheader("🔄 Loading Models...")

for model_file, display_name in MODELS.items():
    model_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(model_path):
        try:
            loaded_models[display_name] = joblib.load(model_path)
            st.success(f"✅ Loaded: {display_name}")
        except Exception as e:
            st.error(f"❌ Failed to load {model_file}: {e}")
    else:
        st.warning(f"⚠️ Model file not found: {model_file}")

if not loaded_models:
    st.error("❌ No models loaded successfully.")
    st.stop()

st.success(f"✅ Successfully loaded {len(loaded_models)} models!")

# ==============================
# 📊 Load Data
# ==============================
data = pd.read_csv(DATA_PATH)

st.subheader("📄 Latest Available Data")
st.dataframe(data.tail(5))

# ==============================
# 🧠 Make Predictions
# ==============================
st.subheader("🔮 Model Predictions")

latest_features = data.iloc[-1:].copy()
latest_features_scaled = pd.DataFrame(
    scaler.transform(latest_features),
    columns=latest_features.columns
)

predictions_data = []
up_votes = 0
down_votes = 0
up_probabilities = []

col1, col2 = st.columns(2)

for idx, (model_name, model) in enumerate(loaded_models.items()):
    expected_features = model.n_features_in_

    if latest_features_scaled.shape[1] != expected_features:
        st.warning(f"⚠️ Feature mismatch for {model_name}")
        continue

    prediction = model.predict(latest_features_scaled)[0]
    proba = model.predict_proba(latest_features_scaled)[0]

    up_probabilities.append(proba[1])

    if prediction == 1:
        up_votes += 1
    else:
        down_votes += 1

    predictions_data.append({
        "Model": model_name,
        "Prediction": "📈 UP" if prediction == 1 else "📉 DOWN",
        "Confidence": f"{float(max(proba)) * 100:.2f}%",
        "Up %": f"{float(proba[1]) * 100:.2f}%",
        "Down %": f"{float(proba[0]) * 100:.2f}%"
    })

    with col1 if idx % 2 == 0 else col2:
        st.markdown(f"### {model_name}")
        if prediction == 1:
            st.success("📈 UP")
        else:
            st.error("📉 DOWN")
        st.progress(float(proba[1]))

# ==============================
# 📊 Summary
# ==============================
st.subheader("📊 Ensemble Summary")

total_models = up_votes + down_votes
avg_up_probability = sum(up_probabilities) / len(up_probabilities) * 100 if up_probabilities else 0

majority_direction = "UP 📈" if up_votes > down_votes else "DOWN 📉"
majority_percentage = (max(up_votes, down_votes) / total_models) * 100 if total_models > 0 else 0

st.metric("📈 UP Predictions", up_votes)
st.metric("📉 DOWN Predictions", down_votes)
st.metric("🏆 Majority Vote", majority_direction)
st.metric("📊 Majority Confidence", f"{majority_percentage:.2f}%")
st.metric("📉 Avg UP Probability", f"{avg_up_probability:.2f}%")

if predictions_data:
    predictions_df = pd.DataFrame(predictions_data)
    st.dataframe(predictions_df, width="stretch")

# ==============================
# ℹ️ Footer
# ==============================
st.divider()
st.caption("🚀 **Developed by AKSHAT MISHRA** | Models: 7 ML Algorithms | Data: web_feed.csv")
st.caption("⚠️ Disclaimer: This is for educational purposes only. Not financial advice.")
