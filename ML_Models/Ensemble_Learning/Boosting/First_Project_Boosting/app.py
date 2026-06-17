from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model & Encoders
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_xgb_regressor.pkl")

sleep_quality_le = joblib.load(
    BASE_DIR / "sleep_quality_label_encoder.pkl"
)

study_method_le = joblib.load(
    BASE_DIR / "study_method_label_encoder.pkl"
)

facility_rating_le = joblib.load(
    BASE_DIR / "facility_rating_label_encoder.pkl"
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Exam Score Prediction",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Exam Score Prediction using XGBoost")
st.markdown("Enter student details below to predict the exam score.")

# Numerical Inputs
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

class_attendance = st.number_input(
    "Class Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

# Categorical Inputs
sleep_quality = st.selectbox(
    "Sleep Quality",
    sleep_quality_le.classes_
)

study_method = st.selectbox(
    "Study Method",
    study_method_le.classes_
)

facility_rating = st.selectbox(
    "Facility Rating",
    facility_rating_le.classes_
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Exam Score"):

    sleep_quality_encoded = sleep_quality_le.transform(
        [sleep_quality]
    )[0]

    study_method_encoded = study_method_le.transform(
        [study_method]
    )[0]

    facility_rating_encoded = facility_rating_le.transform(
        [facility_rating]
    )[0]

    input_df = pd.DataFrame({
        "study_hours": [study_hours],
        "class_attendance": [class_attendance],
        "sleep_hours": [sleep_hours],
        "sleep_quality": [sleep_quality_encoded],
        "study_method": [study_method_encoded],
        "facility_rating": [facility_rating_encoded]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & XGBoost")