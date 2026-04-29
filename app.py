import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================
# 🖥️ Page Config
# ==============================
st.set_page_config(page_title="Transport Delay Predictor", layout="wide")
st.title("🚍 Public Transport Delay Prediction")
st.markdown("Predict delays using weather and event conditions")

# ==============================
# 📦 Load Model + Encoders
# ==============================
model_path = "outputs/model.pkl"
encoders_path = "outputs/encoders.pkl"
features_path = "outputs/feature_columns.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model not found. Run main.py first.")
    st.stop()

if not os.path.exists(encoders_path):
    st.error("❌ Encoders not found. Run main.py first.")
    st.stop()

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# Prefer saved feature list; fallback to model.feature_names_in_
if os.path.exists(features_path):
    feature_cols = joblib.load(features_path)
else:
    feature_cols = list(getattr(model, "feature_names_in_", []))

if not feature_cols:
    st.error("❌ Feature list not found. Save feature_columns.pkl during training.")
    st.stop()

# ==============================
# 🔧 Helpers
# ==============================
def encode_or_default(le, value, default=0):
    if value in le.classes_:
        return le.transform([value])[0]
    return default

def is_categorical(col):
    return col in label_encoders

# ==============================
# 🎛 Sidebar Inputs
# ==============================
st.sidebar.header("🔧 Input Parameters")

# Build a full row with defaults
input_row = {col: 0 for col in feature_cols}

# Create controls dynamically
for col in feature_cols:
    if is_categorical(col):
        le = label_encoders[col]
        options = list(le.classes_)
        choice = st.sidebar.selectbox(f"{col}", options)
        input_row[col] = encode_or_default(le, choice)
    else:
        # numeric input
        input_row[col] = st.sidebar.number_input(f"{col}", value=0.0)

# Create input dataframe
input_data = pd.DataFrame([input_row])

# ==============================
# 🔮 Prediction
# ==============================
prediction = model.predict(input_data)[0]

# ==============================
# 📊 Feature Importance
# ==============================
st.subheader("📊 Feature Importance")
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    features = input_data.columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ==============================
# ✅ Output
# ==============================
st.subheader("📊 Prediction Result")
st.metric(label="Estimated Delay (minutes)", value=f"{prediction:.2f}")

# Show input data
st.subheader("📥 Input Data")
st.write(input_data)

# ==============================
# 📂 Optional Upload
# ==============================
st.subheader("📂 Upload Dataset for Analysis")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    st.subheader("📈 Delay Distribution")
    if "delay_minutes" in df.columns:
        st.bar_chart(df["delay_minutes"].value_counts())
    else:
        st.warning("Column 'delay_minutes' not found.")