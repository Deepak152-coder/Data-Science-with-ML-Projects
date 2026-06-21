import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# ---------------- PAGE ----------------
st.set_page_config(page_title="Smart Parking AI", layout="wide")

st.title("🚗 Smart Parking AI System")
st.subheader("Upload Vehicle Image → Detect Type → Calculate Parking Charges")

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# ---------------- RATES ----------------
rates = {
    "2 Wheeler": 10,
    "Car": 30,
    "Auto": 20,
    "Bus": 70,
    "Truck": 100,
    "Special Category": 15
}

# ---------------- DETECTION ----------------
def detect_vehicle(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]

    labels = [item[1].lower() for item in decoded]

    for label in labels:
        if "motorcycle" in label or "moped" in label:
            return "2 Wheeler"
        elif "car" in label or "cab" in label:
            return "Car"
        elif "bus" in label:
            return "Bus"
        elif "truck" in label or "trailer" in label:
            return "Truck"
        elif "rickshaw" in label:
            return "Auto"

    return "Special Category"

# ---------------- SIDEBAR ----------------
st.sidebar.header("Parking Charges")
for vehicle, price in rates.items():
    st.sidebar.write(f"{vehicle}: ₹{price}/hour")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    vehicle_type = detect_vehicle(image)

    with col2:
        st.success(f"Detected Vehicle: {vehicle_type}")
 
        hours = st.number_input("Parking Hours", min_value=1, max_value=24, value=1)

        total = rates[vehicle_type] * hours

        st.info(f"Rate: ₹{rates[vehicle_type]}/hour")
        st.warning(f"Total Charge: ₹{total}")

        if st.button("Generate Receipt"):
            st.balloons()
            st.write("### Receipt")
            st.write(f"Vehicle Type: {vehicle_type}")
            st.write(f"Hours: {hours}")
            st.write(f"Total Charge: ₹{total}")

st.markdown("---")
st.caption("AI Smart Parking Project by Deepak, Prashant, Govind and Sarthak")