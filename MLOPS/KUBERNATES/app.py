import os
import requests
import streamlit as st
from dotenv import load_dotenv

import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(".env")

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Weather Predictor",
    page_icon="🌦️",
    layout="centered"
)

# -----------------------------
# Custom CSS — Blue Glassmorphism Theme
# -----------------------------
st.markdown(
    """
    <style>
    /* App background — deep blue gradient */
    .stApp {
        background: linear-gradient(160deg, #03060f 0%, #061229 35%, #0a1f3d 65%, #0d2c52 100%);
        background-attachment: fixed;
    }

    /* Hide default streamlit chrome */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Title styling */
    .app-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7fd8ff, #4ea1ff, #8a7bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px;
    }

    .app-subtitle {
        text-align: center;
        color: #9fb3d1;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    /* Input field glass styling */
    div[data-testid="stTextInput"] input {
        background: rgba(20, 35, 65, 0.55) !important;
        border: 1px solid rgba(120, 170, 255, 0.35) !important;
        border-radius: 12px !important;
        color: #e8f1ff !important;
        padding: 0.7rem 1rem !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #6f87ad !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border: 1px solid #4ea1ff !important;
        box-shadow: 0 0 0 2px rgba(78, 161, 255, 0.25) !important;
    }

    /* Button styling */
    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #1e5fae, #4ea1ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 18px rgba(78, 161, 255, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(78, 161, 255, 0.55);
        color: white;
        border: none;
    }

    /* Glass card container for results */
    .glass-card {
        background: rgba(20, 35, 65, 0.45);
        border: 1px solid rgba(120, 170, 255, 0.25);
        border-radius: 18px;
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 10, 40, 0.45);
    }

    .location-text {
        text-align: center;
        color: #cfe4ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background: rgba(10, 22, 45, 0.55);
        border: 1px solid rgba(120, 170, 255, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.5rem;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        color: #8fb3e0 !important;
        justify-content: center;
    }
    div[data-testid="stMetricValue"] {
        color: #e8f1ff !important;
        font-size: 1.4rem !important;
    }

    /* Condition box */
    .condition-box {
        text-align: center;
        background: rgba(78, 161, 255, 0.12);
        border: 1px solid rgba(78, 161, 255, 0.3);
        border-radius: 14px;
        padding: 0.9rem;
        color: #cfe4ff;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    .icon-wrap {
        display: flex;
        justify-content: center;
        margin-top: 0.5rem;
    }

    /* Warnings/errors/success — tint to match theme */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="app-title">🌦️ Weather Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Enter a city name to get live weather conditions</div>', unsafe_allow_html=True)

city = st.text_input(
    "City",
    placeholder="e.g. Varanasi, New Delhi, Tokyo",
    label_visibility="collapsed"
)

if st.button("Get Weather"):

    if not city.strip():
        st.warning("Please enter a city name.")

    elif API_KEY is None:
        st.error("API Key not found. Check your .env file.")

    else:

        url = (
            f"http://api.weatherapi.com/v1/current.json?"
            f"key={API_KEY}&q={city}&aqi=yes"
        )

        with st.spinner("Fetching weather data..."):
            response = requests.get(url)

        if response.status_code == 200:

            data = response.json()

            location = data["location"]
            current = data["current"]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            st.markdown(
                f'<div class="location-text">📍 {location["name"]}, {location["region"]}, {location["country"]}</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🌡 Temperature", f"{current['temp_c']} °C")

            with col2:
                st.metric("🥵 Feels Like", f"{current['feelslike_c']} °C")

            with col3:
                st.metric("💧 Humidity", f"{current['humidity']} %")

            st.write("")

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric("🌬 Wind", f"{current['wind_kph']} km/h")

            with col5:
                st.metric("🧭 Pressure", f"{current['pressure_mb']} mb")

            with col6:
                st.metric("👁 Visibility", f"{current['vis_km']} km")

            st.markdown('<div class="icon-wrap">', unsafe_allow_html=True)
            st.image("https:" + current["condition"]["icon"], width=100)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="condition-box">{current["condition"]["text"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error("Unable to fetch weather information.")
            st.write(response.status_code)
            st.json(response.json())