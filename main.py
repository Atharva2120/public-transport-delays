import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 📥 Load Data
# ==============================
print("📥 Loading data...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "public_transport_delays.csv")  # <-- change filename if needed

df = pd.read_csv(file_path)

# ==============================
# 🧹 Data Cleaning
# ==============================
print("🧹 Cleaning data...")

df = df.dropna()

# ==============================
# ⚙️ Feature Engineering
# ==============================
print("⚙️ Feature engineering...")

target = "delayed"

# 🚨 Remove leakage columns
leakage_cols = [
    "actual_arrival_delay_min",
    "actual_departure_delay_min"
]

# Optional drops (IDs / non-useful)
drop_cols = [
    "trip_id",
    "date",
    "time"
]

df = df.drop(columns=leakage_cols + drop_cols, errors="ignore")

# ==============================
# 🔤 Encode Categorical Columns
# ==============================
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ==============================
# 🎯 Split Features & Target
# ==============================
X = df.drop(columns=[target])
y = df[target]

# ==============================
# ✂️ Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 🤖 Train Model
# ==============================
print("🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 📊 Evaluate Model
# ==============================
print("📊 Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 💾 Save Model
# ==============================
print("💾 Saving model...")

os.makedirs("outputs", exist_ok=True)

joblib.dump(model, "outputs/model.pkl")
joblib.dump(label_encoders, "outputs/encoders.pkl")

print("✅ Done! Model saved in outputs/")