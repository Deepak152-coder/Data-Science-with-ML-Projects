import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("naive_bayes_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details below:")

# Numerical Inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, value=5)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=700)

# Binary Feature
previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    ["No", "Yes"]
)

# Gender
person_gender_male = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

# Education
education = st.selectbox(
    "Education",
    [
        "Bachelor",
        "Doctorate",
        "High School",
        "Master",
        "Other"
    ]
)

# Home Ownership
home_ownership = st.selectbox(
    "Home Ownership",
    [
        "MORTGAGE",
        "OTHER",
        "OWN",
        "RENT"
    ]
)

# Loan Intent
loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE"
    ]
)

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0

    # Numerical Features
    input_data.at[0, "person_age"] = person_age
    input_data.at[0, "person_income"] = person_income
    input_data.at[0, "person_emp_exp"] = person_emp_exp
    input_data.at[0, "loan_amnt"] = loan_amnt
    input_data.at[0, "loan_int_rate"] = loan_int_rate
    input_data.at[0, "loan_percent_income"] = loan_percent_income
    input_data.at[0, "cb_person_cred_hist_length"] = cb_person_cred_hist_length
    input_data.at[0, "credit_score"] = credit_score

    # Binary
    input_data.at[0, "previous_loan_defaults_on_file"] = (
        1 if previous_loan_defaults_on_file == "Yes" else 0
    )

    # Gender
    input_data.at[0, "person_gender_male"] = (
        1 if person_gender_male == "Male" else 0
    )

    # Education
    edu_col = f"person_education_{education}"
    if edu_col in input_data.columns:
        input_data.at[0, edu_col] = 1

    # Home Ownership
    home_col = f"person_home_ownership_{home_ownership}"
    if home_col in input_data.columns:
        input_data.at[0, home_col] = 1

    # Loan Intent
    intent_col = f"loan_intent_{loan_intent}"
    if intent_col in input_data.columns:
        input_data.at[0, intent_col] = 1

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")