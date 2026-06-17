import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("knn_significant_features_model.joblib")
scaler = joblib.load("scaler.joblib")

# Page settings
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student GPA Prediction System")

st.markdown("Enter student details below.")

# Inputs

study_time = st.number_input(
    "Study Time Weekly (Hours)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)

tutoring = st.selectbox(
    "Tutoring",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

parental_support = st.selectbox(
    "Parental Support",
    [0, 1, 2, 3, 4]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

sports = st.selectbox(
    "Sports",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

music = st.selectbox(
    "Music",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

grade_class = st.selectbox(
    "Grade Class",
    [0, 1, 2, 3, 4]
)

# Prediction

if st.button("Predict GPA"):

    features = np.array([[
        study_time,
        absences,
        tutoring,
        parental_support,
        extracurricular,
        sports,
        music,
        grade_class
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    st.success(f"Predicted GPA: {prediction[0]:.2f}")