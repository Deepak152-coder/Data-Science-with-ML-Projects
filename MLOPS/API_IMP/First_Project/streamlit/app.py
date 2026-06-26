import streamlit as st
import requests

# FastAPI URL
API_URL = "https://data-science-with-ml-projects-1.onrender.com/predict"
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Delivery Time Prediction")
st.write("Enter the delivery details below and click **Predict**.")

distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.5
)

prep_time = st.number_input(
    "Preparation Time (min)",
    min_value=0.0,
    value=20.0,
    step=1.0
)

weather = st.selectbox(
    "Weather",
    ["Sunny", "Cloudy", "Rainy", "Foggy", "Stormy"]
)

traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

if st.button("Predict Delivery Time", use_container_width=True):

    payload = {
        "Distance_km": distance,
        "Preparation_Time_min": prep_time,
        "Weather": weather,
        "Traffic_Level": traffic
    }

    try:

        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:

            prediction = response.json()["Predicted_Delivery_Time"]

            st.success("Prediction Successful! ✅")
            st.metric(
                "Estimated Delivery Time",
                f"{prediction:.2f} minutes"
            )

        else:

            st.error(f"API Error ({response.status_code})")
            st.code(response.text)

    except requests.exceptions.ConnectionError as e:
        st.error("❌ Cannot connect to FastAPI.")
        st.code(str(e))

    except Exception as e:
        st.error("Unexpected Error")
        st.code(str(e))