import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Salary Prediction App")
st.write("Predict Salary based on Years of Experience")

st.divider()

# User Input
years_exp = st.slider(
    "Select Years of Experience",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):

    input_data = np.array([[years_exp]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Salary: ₹ {prediction[0]:,.2f}"
    )

st.divider()

# Show model behavior
st.subheader("Model Predictions (0-10 Years)")

years = np.arange(0, 10.5, 0.5).reshape(-1, 1)

years_scaled = scaler.transform(years)

predictions = model.predict(years_scaled)

chart_df = pd.DataFrame({
    "YearsExperience": years.flatten(),
    "PredictedSalary": predictions
})

st.line_chart(
    chart_df.set_index("YearsExperience")
)

st.divider()

st.info("Model: KNN Regressor")