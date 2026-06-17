import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("decision_tree_model.joblib")

# Page Config
st.set_page_config(
    page_title="Drug Prediction System",
    page_icon="💊",
    layout="centered"
)

# Title
st.title("💊 Drug Recommendation System")
st.markdown(
    "Predict the most suitable drug based on patient information."
)

st.divider()

# User Inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["F", "M"]
)

bp = st.selectbox(
    "Blood Pressure",
    ["HIGH", "LOW", "NORMAL"]
)

cholesterol = st.selectbox(
    "Cholesterol",
    ["HIGH", "NORMAL"]
)

na_to_k = st.number_input(
    "Na_to_K Ratio",
    min_value=0.0,
    max_value=50.0,
    value=15.0
)

# Predict Button
if st.button("🔍 Predict Drug"):

    # Encoding exactly as training
    sex_map = {
        "F": 0,
        "M": 1
    }

    bp_map = {
        "HIGH": 0,
        "LOW": 1,
        "NORMAL": 2
    }

    cholesterol_map = {
        "HIGH": 0,
        "NORMAL": 1
    }

    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_map[sex]],
        "BP": [bp_map[bp]],
        "Cholesterol": [cholesterol_map[cholesterol]],
        "Na_to_K": [na_to_k]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"💊 Recommended Drug: {prediction}")

    st.subheader("📋 Patient Information")

    st.dataframe(
        pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "BP": [bp],
            "Cholesterol": [cholesterol],
            "Na_to_K": [na_to_k]
        }),
        use_container_width=True
    )

st.divider()

st.info(
    "This application uses a Decision Tree Classifier trained on the Drug200 dataset."
)

st.caption(
    "Built with Streamlit | Machine Learning Project"
)