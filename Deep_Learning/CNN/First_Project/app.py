import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    color: #2E8B57;
    font-size: 3rem;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 1.1rem;
    margin-bottom: 20px;
}

.result-box {
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.keras")

model = load_model()

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    '<div class="title">🍎 AI Fruit Freshness Detector 🍌</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning CNN Model for Fresh vs Rotten Fruit Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:

    st.header("📊 Model Information")

    st.success("Validation Accuracy: 98.17%")

    st.markdown("""
    ### 🧠 Model Details
    - CNN Architecture
    - TensorFlow / Keras
    - Binary Classification
    - Image Size: 224 × 224

    ### 🎯 Classes
    - Fresh
    - Rotten

    ### 📸 Supported Formats
    - JPG
    - JPEG
    - PNG
    """)

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ----------------------------
    # PREPROCESSING
    # ----------------------------
    img = image.resize((224, 224))

    img_array = np.array(img)

    # Same preprocessing used during training
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    prediction = model.predict(img_array, verbose=0)[0][0]

    fresh_prob = (1 - prediction) * 100
    rotten_prob = prediction * 100

    if prediction >= 0.5:
        result = "🍂 Rotten"
        confidence = rotten_prob
    else:
        result = "🍏 Fresh"
        confidence = fresh_prob

    with col2:

        st.subheader("🔍 Prediction Result")

        if prediction >= 0.5:
            st.error(result)
        else:
            st.success(result)

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    st.divider()

    # ----------------------------
    # PROBABILITY SECTION
    # ----------------------------
    st.subheader("📈 Prediction Probabilities")

    st.write(f"🍏 Fresh : {fresh_prob:.2f}%")
    st.progress(float(fresh_prob / 100))

    st.write(f"🍂 Rotten : {rotten_prob:.2f}%")
    st.progress(float(rotten_prob / 100))

    st.divider()

    # ----------------------------
    # FINAL VERDICT
    # ----------------------------
    st.subheader("🏆 Final Verdict")

    if prediction >= 0.5:
        st.error(
            f"⚠️ The fruit appears to be ROTTEN with {rotten_prob:.2f}% confidence."
        )
    else:
        st.success(
            f"✅ The fruit appears to be FRESH with {fresh_prob:.2f}% confidence."
        )

# ----------------------------
# FOOTER
# ----------------------------
st.divider()

st.markdown(
    """
    <div class="footer">
        🚀 Built with TensorFlow, Keras & Streamlit <br>
        CNN-Based Fruit Freshness Detection System
    </div>
    """,
    unsafe_allow_html=True
)