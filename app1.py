import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import random
import time

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="NEURODIGIT", page_icon="🔢", layout="centered")

# ---------------------------
# Custom CSS with Enhanced User-Friendly Background
# ---------------------------
page_bg = """
<style>
/* Import Google Fonts for better typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    /* Beautiful gradient background with subtle animation */
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
    color: #ffffff;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Animated gradient background */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Enhanced glass morphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    color: #ffffff !important;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* Typography improvements */
h1, h2, h3 {
    color: #ffffff !important;
    text-align: center;
    font-weight: 600;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

h1 {
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
}

p, span, label, .stMarkdown, .css-1offfwp {
    color: #f8f9fa !important;
    font-weight: 400;
}

.css-1cpxqw2 {
    color: #e9ecef !important;
    font-size: 14px;
}

/* Enhanced button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    font-weight: 600;
    border-radius: 15px;
    border: none;
    padding: 0.75rem 2rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* Enhanced info and warning boxes */
.info-box {
    background: rgba(13, 202, 240, 0.2);
    border: 1px solid rgba(13, 202, 240, 0.4);
    border-left: 4px solid #0dcaf0;
    padding: 1rem;
    border-radius: 12px;
    color: #ffffff;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}

.warning-box {
    background: rgba(255, 193, 7, 0.2);
    border: 1px solid rgba(255, 193, 7, 0.4);
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 12px;
    color: #ffffff;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}

/* Radio button styling */
.stRadio > div {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Progress bar styling */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 10px;
}

/* File uploader styling */
.stFileUploader > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    border: 2px dashed rgba(255, 255, 255, 0.3);
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: rgba(255, 255, 255, 0.5);
    background: rgba(255, 255, 255, 0.15);
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
}

/* Camera input styling */
.stCameraInput > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
}

/* Image styling */
.stImage {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# Enhanced App Header
# ---------------------------
st.markdown(
    """
    <div class="glass-card" style="text-align:center;">
        <h1>🔢 NeuroDigit</h1>
        <p style="font-size:20px; margin-bottom: 1rem; font-weight: 300;">
            Recognize handwritten digits instantly with deep learning
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <div style="font-size: 14px; opacity: 0.9;">AI Powered</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="font-size: 14px; opacity: 0.9;">Real-time</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 14px; opacity: 0.9;">Accurate</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Preprocessing & Prediction
# ---------------------------
def preprocess_image(img):
    img_gray = img.convert("L") if img.mode != "L" else img
    img_array = np.array(img_gray)
    _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    inverted_img = cv2.bitwise_not(binary_img)
    resized_img = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)
    norm_img = resized_img / 255.0
    return norm_img

def predict_digit(img_array):
    input_img = img_array.reshape(1, 28, 28, 1).astype("float32")
    prediction = model.predict(input_img, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_digit, confidence

# ---------------------------
# Tips Section
# ---------------------------
st.markdown(
    """
    <div class="glass-card">
        <h3>💡 Tips for Best Results</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="display: flex; align-items: center; padding: 0.75rem; background: rgba(255, 255, 255, 0.1); border-radius: 12px; border-left: 4px solid #00c6ff;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">✍️</div>
                <div>
                    <strong>Write Clearly</strong><br>
                    <span style="opacity: 0.9; font-size: 14px;">Use a dark pen on white paper for best contrast</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; padding: 0.75rem; background: rgba(255, 255, 255, 0.1); border-radius: 12px; border-left: 4px solid #28a745;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">📸</div>
                <div>
                    <strong>Hold Steady</strong><br>
                    <span style="opacity: 0.9; font-size: 14px;">Keep the digit centered and capture a clear image</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; padding: 0.75rem; background: rgba(255, 255, 255, 0.1); border-radius: 12px; border-left: 4px solid #ffc107;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">🎲</div>
                <div>
                    <strong>Try Samples</strong><br>
                    <span style="opacity: 0.9; font-size: 14px;">Use sample images below for quick testing</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Navigation
# ---------------------------
choice = st.radio(
    "Choose Input Method:",
    ["📷 Camera", "📤 Upload File", "🎲 Sample Data", "🎞️ Slideshow"],
    horizontal=True
)

image = None

if choice == "📷 Camera":
    st.markdown(
        """
        <div class="info-box">
            📸 <strong>Camera Tips:</strong> Make sure your digit is well-lit and fills most of the frame. Hold your device steady for a clear capture.
        </div>
        """,
        unsafe_allow_html=True
    )
    camera_img = st.camera_input("Capture your handwritten digit")
    if camera_img:
        image = Image.open(camera_img)

elif choice == "📤 Upload File":
    st.markdown(
        """
        <div class="info-box">
            📤 <strong>Upload Tips:</strong> Choose a clear image with good contrast. PNG and JPG formats work best.
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)

elif choice == "🎲 Sample Data":
    st.markdown(
        """
        <div class="info-box">
            🎲 <strong>Sample Tips:</strong> These are real handwritten digits from the MNIST dataset. Perfect for testing the AI model's accuracy!
        </div>
        """,
        unsafe_allow_html=True
    )
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    sample_indices = random.sample(range(len(x_test)), 10)
    labels = [y_test[i] for i in sample_indices]
    st.markdown("### 🔍 Sample Previews")
    preview_imgs = []
    for i in range(10):
        img = x_test[sample_indices[i]]
        img_inverted = cv2.bitwise_not(img)
        preview_imgs.append(Image.fromarray(img_inverted))
    st.image(preview_imgs, caption=[f"{labels[i]}" for i in range(10)], width=80)
    sample_choice = st.selectbox("Choose a sample digit:", [f"Digit {labels[i]} (Sample {i+1})" for i in range(10)])
    if st.button("Load Sample"):
        idx = int(sample_choice.split("Sample ")[1].replace(")", "")) - 1
        img = x_test[sample_indices[idx]]
        label = y_test[sample_indices[idx]]
        img_inverted = cv2.bitwise_not(img)
        image = Image.fromarray(img_inverted)
        st.markdown(f"<div class='info-box'>Sample digit <b>{label}</b> loaded.</div>", unsafe_allow_html=True)

elif choice == "🎞️ Slideshow":
    st.markdown(
        """
        <div class="info-box">
            🎞️ <strong>Slideshow Tips:</strong> Sit back and watch the AI automatically predict 5 random digits. Great for demonstrating the model's capabilities!
        </div>
        """,
        unsafe_allow_html=True
    )
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    st.markdown("### 🎞️ Slideshow Mode - Auto Testing 5 Samples")
    slideshow_area = st.empty()
    for i in range(5):
        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        label = y_test[idx]
        img_inverted = cv2.bitwise_not(img)
        processed_img = preprocess_image(Image.fromarray(img_inverted))
        digit, conf = predict_digit(processed_img)
        with slideshow_area.container():
            st.image(img_inverted, caption=f"True: {label}", width=150)
            st.markdown(f"**Predicted: {digit} | Confidence: {conf:.2f}**")
            st.progress(int(conf * 100))
        time.sleep(2)

# ---------------------------
# Display Prediction
# ---------------------------
if image is not None and choice != "🎞️ Slideshow":
    processed_img = preprocess_image(image)
    st.markdown("### 🖼️ Processed Image")
    st.image(processed_img, caption="Background = Black, Digit = White", width=200, clamp=True)
    digit, conf = predict_digit(processed_img)
    if conf <= 0.5:
        st.markdown("<div class='warning-box'>⚠️ Sorry, I cannot recognize the number clearly.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"### ✅ Predicted Digit: **{digit}**")
        st.progress(int(conf * 100))
        st.markdown(f"**Confidence:** {conf:.2f}")
    