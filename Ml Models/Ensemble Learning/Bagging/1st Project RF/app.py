import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load model and feature names
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "best_random_forest_model.pkl")
)

model_features = joblib.load(
    os.path.join(BASE_DIR, "model_features.pkl")
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Food Delivery Time Predictor")
st.write("Enter delivery details and predict delivery time.")

# -----------------------------
# User Inputs
# -----------------------------
distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

prep_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=0,
    value=20,
    step=1
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Rainy", "Foggy", "Snowy", "Windy"]
)

traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Delivery Time"):

    # Create empty dataframe with all model columns
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    # Fill numerical columns
    if "Distance_km" in input_df.columns:
        input_df["Distance_km"] = distance

    if "Preparation_Time_min" in input_df.columns:
        input_df["Preparation_Time_min"] = prep_time

    # Fill Weather dummy column
    weather_col = f"Weather_{weather}"
    if weather_col in input_df.columns:
        input_df[weather_col] = 1

    # Fill Traffic dummy column
    traffic_col = f"Traffic_Level_{traffic}"
    if traffic_col in input_df.columns:
        input_df[traffic_col] = 1

    # Make prediction
    prediction = model.predict(input_df)[0]

    st.success(
        f"⏱ Estimated Delivery Time: {prediction:.2f} minutes"
    )

    st.balloons()