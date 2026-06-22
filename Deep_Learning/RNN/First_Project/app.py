import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#4F8BF9;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:20px;
}

.metric-card {
    padding:15px;
    border-radius:15px;
    background-color:#f5f7ff;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

.result-spam {
    background:#ffebee;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#d32f2f;
    font-size:28px;
    font-weight:bold;
}

.result-ham {
    background:#e8f5e9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#2e7d32;
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
@st.cache_resource
def load_artifacts():
    model = load_model("gru_spam_classifier.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_artifacts()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_email(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    pred = float(model.predict(padded, verbose=0)[0][0])

    if pred >= 0.5:
        label = "Spam"
        confidence = pred * 100
    else:
        label = "Ham"
        confidence = (1 - pred) * 100

    return label, confidence, pred

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">📧 Email Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning based Spam Detection using GRU Neural Networks</div>',
    unsafe_allow_html=True
)

# =========================
# PROJECT STATS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "GRU")

with col2:
    st.metric("Accuracy", "97.33%")

with col3:
    st.metric("Vocabulary", "5000")

st.markdown("---")

# =========================
# INPUT
# =========================
st.subheader("✉️ Enter Email Content")

email_text = st.text_area(
    "",
    height=220,
    placeholder="Paste the email content here..."
)

# =========================
# BUTTON
# =========================
predict_btn = st.button(
    "🔍 Analyze Email",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================
if predict_btn:

    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:

        label, confidence, spam_prob = predict_email(email_text)

        st.markdown("---")

        if label == "Spam":

            st.markdown(
                f"""
                <div class="result-spam">
                🚨 SPAM EMAIL DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-ham">
                ✅ LEGITIMATE EMAIL
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        with c2:
            st.metric(
                "Spam Probability",
                f"{spam_prob*100:.2f}%"
            )

        st.progress(float(confidence) / 100)

# =========================
# EXAMPLES
# =========================
st.markdown("---")

with st.expander("🚨 Example Spam Email"):
    st.code("""
Congratulations!

You have won a FREE iPhone 16.

Click below to claim your prize immediately.

Limited time offer.
""")

with st.expander("✅ Example Legitimate Email"):
    st.code("""
Hello Team,

Please find the attached project report.

Let me know if any modifications are required.

Regards,
Deepak
""")

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption(
    "Built with TensorFlow • GRU • NLP • Streamlit"
)