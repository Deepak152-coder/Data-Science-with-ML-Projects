import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🧠 Personality Prediction System")

features = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "listening_skill",
    "empathy",
    "creativity",
    "organization",
    "leadership",
    "risk_taking",
    "public_speaking_comfort",
    "curiosity",
    "routine_preference",
    "excitement_seeking",
    "friendliness",
    "emotional_stability",
    "planning",
    "spontaneity",
    "adventurousness",
    "reading_habit",
    "sports_interest",
    "online_social_usage",
    "travel_desire",
    "gadget_usage",
    "work_style_collaborative",
    "decision_speed",
    "stress_handling"
]

user_input = {}

for feature in features:
    user_input[feature] = st.slider(
        feature,
        min_value=0,
        max_value=100,
        value=50
    )

if st.button("Predict Personality"):

    input_df = pd.DataFrame([user_input])

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)

    personality = encoder.inverse_transform(prediction)

    st.success(
        f"Predicted Personality: {personality[0]}"
    )