# Public Transport Delay Prediction 🚍

A machine learning project that predicts whether a public transport trip will be delayed based on weather, traffic, schedule, and event conditions. Includes a training script and a Streamlit web app for interactive predictions.

---

## ✨ Features
- Train a RandomForest model using historical delay data
- Encode categorical features automatically
- Interactive Streamlit UI for prediction
- Feature-importance visualization
- Optional CSV upload for quick analysis
- Notebook for delay-duration regression modeling

---

---

## 🗂 Project Structure
```
public-transport-delays/
├── app.py                 # Streamlit app
├── main.py                # Model training script
├── data/                  # Dataset folder (public_transport_delays.csv)
├── notebooks/             # Jupyter notebooks
├── outputs/               # Saved model + encoders
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description
The dataset contains trip, schedule, weather, and event signals used to predict delays.

**Common columns (sample):**
- `trip_id`, `date`, `time`, `transport_mode`, `route_id`
- `origin_station`, `destination_station`
- `scheduled_departure`, `scheduled_arrival`
- `actual_departure`, `actual_arrival`
- `weather_condition`, `temperature_c`, `humidity_percent`, `wind_speed_kmh`, `precipitation_mm`
- `event_type`, `event_attendance_est`
- `traffic_congestion_index`, `holiday`, `peak_hour`, `weekday`, `season`

**Target column:**
- `delayed` (classification label)

**Leakage columns removed during training:**
- `actual_arrival_delay_min`, `actual_departure_delay_min`

---

## 📓 Notebook: Delay Duration Prediction
A regression notebook is available at:

- `notebooks/delay_duration_model.ipynb`

It trains a RandomForestRegressor pipeline with preprocessing and evaluates MAE/RMSE/R². The trained model is saved to:

- `outputs/delay_duration_model.pkl`

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

Make sure your dataset is in:
```
data/public_transport_delays.csv
```

Then run:

```bash
python main.py
```

Outputs:
```
outputs/model.pkl
outputs/encoders.pkl
```

---

## 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Click **New app** and select this repo.
4. Set the entry point to `app.py`.
5. Deploy.

---

## 📌 Notes
- Ensure the features at prediction time match the training features.
- If you change the dataset schema, retrain the model.

---

## 👤 Author
**Atharva2120**
