import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("decision_tree_regressor_model.joblib")
le = joblib.load("label_encoder.joblib")

# Page Config
st.set_page_config(
    page_title="Car Selling Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Selling Price Prediction")
st.write("Enter the car details to predict its selling price.")

# Inputs
car_name = st.selectbox(
    "Car Name",
    le.classes_
)

year = st.slider(
    "Manufacturing Year",
    min_value=2000,
    max_value=2025,
    value=2018
)

present_price = st.number_input(
    "Present Price (Lakhs ₹)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

# Encoding
car_name_encoded = le.transform([car_name])[0]

fuel_type_map = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
}

seller_type_map = {
    "Dealer": 0,
    "Individual": 1
}

transmission_map = {
    "Manual": 0,
    "Automatic": 1
}

fuel_type_encoded = fuel_type_map[fuel_type]
seller_type_encoded = seller_type_map[seller_type]
transmission_encoded = transmission_map[transmission]

# Prediction
if st.button("Predict Selling Price"):

    input_data = pd.DataFrame(
        [[
            car_name_encoded,
            year,
            present_price,
            kms_driven,
            fuel_type_encoded,
            seller_type_encoded,
            transmission_encoded,
            owner
        ]],
        columns=[
            'Car_Name',
            'Year',
            'Present_Price',
            'Kms_Driven',
            'Fuel_Type',
            'Seller_Type',
            'Transmission',
            'Owner'
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Predicted Selling Price: ₹ {prediction:.2f} Lakhs"
    )

    st.balloons()