import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("decision_tree_regressor_model.joblib")
model_features = joblib.load("model_features.joblib")

st.set_page_config(
    page_title="USA Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 USA Housing Price Prediction")
st.write("Enter housing details below.")

# Inputs
income = st.number_input(
    "Average Area Income",
    min_value=0.0,
    value=70000.0
)

house_age = st.number_input(
    "Average Area House Age",
    min_value=0.0,
    value=6.0
)

rooms = st.number_input(
    "Average Area Number of Rooms",
    min_value=0.0,
    value=7.0
)

bedrooms = st.number_input(
    "Average Area Number of Bedrooms",
    min_value=0.0,
    value=4.0
)

population = st.number_input(
    "Area Population",
    min_value=0.0,
    value=35000.0
)

if st.button("Predict House Price"):

    input_dict = {
        "Avg. Area Income": income,
        "Avg. Area House Age": house_age,
        "Avg. Area Number of Rooms": rooms,
        "Avg. Area Number of Bedrooms": bedrooms,
        "Area Population": population
    }

    # Create dataframe
    input_data = pd.DataFrame([input_dict])

    # Keep only features used during training
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted House Price: ${prediction:,.2f}")