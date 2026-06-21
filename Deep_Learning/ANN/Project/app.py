import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# =====================================================
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ann_laptop_price_model.keras")

model = load_model()

scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
cat_col = joblib.load("cat_col.pkl")
num_col = joblib.load("num_col.pkl")

df = pd.read_csv("laptop_price.csv", encoding="latin1")

# =====================================================
# HEADER
# =====================================================
st.title("💻 Laptop Price Prediction using ANN")
st.markdown(
    """
    Predict Laptop Prices in **Indian Rupees (₹)** using a trained
    **Artificial Neural Network (ANN)** model.
    """
)

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("⚙️ Laptop Specifications")

company = st.sidebar.selectbox(
    "Company",
    sorted(df["Company"].dropna().unique())
)

typename = st.sidebar.selectbox(
    "Type",
    sorted(df["TypeName"].dropna().unique())
)

screen = st.sidebar.selectbox(
    "Screen Resolution",
    sorted(df["ScreenResolution"].dropna().unique())
)

cpu = st.sidebar.selectbox(
    "CPU",
    sorted(df["Cpu"].dropna().unique())
)

memory = st.sidebar.selectbox(
    "Memory",
    sorted(df["Memory"].dropna().unique())
)

gpu = st.sidebar.selectbox(
    "GPU",
    sorted(df["Gpu"].dropna().unique())
)

osys = st.sidebar.selectbox(
    "Operating System",
    sorted(df["OpSys"].dropna().unique())
)

inches = st.sidebar.number_input(
    "Screen Size (Inches)",
    min_value=10.0,
    max_value=20.0,
    value=15.6,
    step=0.1
)

ram = st.sidebar.number_input(
    "RAM (GB)",
    min_value=2,
    max_value=64,
    value=8,
    step=2
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Predict Laptop Price", use_container_width=True):

    user_input = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "ScreenResolution": screen,
        "Cpu": cpu,
        "Memory": memory,
        "Gpu": gpu,
        "OpSys": osys,
        "Inches": inches,
        "Ram": ram,
        "Weight": weight
    }])

    # One Hot Encoding
    user_input = pd.get_dummies(
        user_input,
        columns=cat_col,
        drop_first=True
    )

    # Match training columns
    user_input = user_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Remove target if present
    if "Price_euros" in user_input.columns:
        user_input = user_input.drop(
            columns=["Price_euros"]
        )

    # Scale numerical columns
    user_input[num_col] = scaler.transform(
        user_input[num_col]
    )

    # Prediction
    prediction = model.predict(
        user_input,
        verbose=0
    )

    price_euro = float(prediction[0][0])

    # Fixed EUR → INR conversion
    EUR_TO_INR = 100

    price_inr = max(price_euro * EUR_TO_INR, 0)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🇮🇳 Predicted Price",
            value=f"₹ {price_inr:,.0f}"
        )

    with col2:
        st.metric(
            label="💶 Price in Euro",
            value=f"€ {price_euro:,.2f}"
        )

    st.success(
        f"Estimated Laptop Price: ₹ {price_inr:,.0f}"
    )

    st.balloons()

# =====================================================
# DATASET / MODEL INFO
# =====================================================
st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Dataset Rows",
        f"{len(df):,}"
    )

with c2:
    st.metric(
        "Features",
        len(feature_columns)
    )

with c3:
    st.metric(
        "Model R² Score",
        "0.835"
    )

st.markdown("---")

st.info(
    """
    **Model Performance**
    
    • R² Score: 0.835  
    • MAE: 188.95  
    • MSE: 83,704.29  
    """
)

st.caption(
    "Built with TensorFlow • Scikit-Learn • Streamlit"
)