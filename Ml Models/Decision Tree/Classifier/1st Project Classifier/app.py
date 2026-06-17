import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Artifacts
# =========================
model = joblib.load("best_decision_tree_model.pkl")
scaler = joblib.load("standard_scaler.pkl")
gender_encoder = joblib.load("label_encoder_gender.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================
# Header
# =========================
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Predict the likelihood of heart disease based on a person's health and lifestyle factors.
    """
)

# =========================
# Sidebar
# =========================
st.sidebar.header("ℹ️ About")

st.sidebar.info(
    """
    This application uses a Decision Tree Classifier
    trained on health and lifestyle data.

    Features used:
    - Age
    - BMI
    - Daily Steps
    - Sleep Hours
    - Water Intake
    - Cholesterol
    - Blood Pressure
    - Smoking Status
    - Alcohol Consumption
    - Family History
    """
)

# =========================
# Input Section
# =========================
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        list(gender_encoder.classes_)
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    daily_steps = st.number_input(
        "Daily Steps",
        min_value=0,
        max_value=50000,
        value=8000
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    water_intake_l = st.number_input(
        "Water Intake (Liters)",
        min_value=0.0,
        max_value=10.0,
        value=2.5
    )

    calories_consumed = st.number_input(
        "Calories Consumed",
        min_value=0,
        max_value=10000,
        value=2200
    )

with col2:

    smoker = st.selectbox(
        "Smoker",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    resting_hr = st.number_input(
        "Resting Heart Rate",
        min_value=30,
        max_value=200,
        value=70
    )

    systolic_bp = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=150,
        value=80
    )

    cholesterol = st.number_input(
        "Cholesterol Level",
        min_value=50,
        max_value=500,
        value=180
    )

    family_history = st.selectbox(
        "Family History",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Heart Disease"):

    gender_encoded = gender_encoder.transform([gender])[0]

    input_df = pd.DataFrame({
        'age': [age],
        'gender': [gender_encoded],
        'bmi': [bmi],
        'daily_steps': [daily_steps],
        'sleep_hours': [sleep_hours],
        'water_intake_l': [water_intake_l],
        'calories_consumed': [calories_consumed],
        'smoker': [smoker],
        'alcohol': [alcohol],
        'resting_hr': [resting_hr],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'cholesterol': [cholesterol],
        'family_history': [family_history]
    })

    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.divider()

    # =====================
    # Result
    # =====================
    if prediction == 1:

        confidence = model.predict_proba(input_scaled)[0][1] * 100

        st.error(
            f"⚠️ High Risk of Heart Disease Detected\n\nConfidence: {confidence:.2f}%"
        )

    else:

        confidence = model.predict_proba(input_scaled)[0][0] * 100

        st.success(
            f"✅ No Significant Risk of Heart Disease Detected\n\nConfidence: {confidence:.2f}%"
        )

    # =====================
    # Summary
    # =====================
    st.subheader("📋 Patient Summary")

    st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit | Decision Tree Classifier | Machine Learning Project")