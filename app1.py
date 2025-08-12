import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
import os

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("✏️ Handwritten Digit Recognition")
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stRadio > div {
        display: flex;
        gap: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Upload or capture an image of a digit (0-9) and get the prediction instantly!")

# ---------------------------
# Input Method
# ---------------------------
input_method = st.radio("📷 Choose Input Method:", ("Camera", "Upload File"))

file = None
if input_method == "Camera":
    file = st.camera_input("Take a photo of your digit")
elif input_method == "Upload File":
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# ---------------------------
# Process & Predict
# ---------------------------
if file is not None:
    img = Image.open(file).convert("L")
    img_resized = img.resize((28, 28), Image.LANCZOS)
    img_inverted = ImageOps.invert(img_resized)
    img_array = np.array(img_inverted) / 255.0
    img_array = img_array.reshape(1, -1).astype(np.float32)

    st.image(img_inverted, caption="🖼 Processed Image (28x28)", width=150)

    if st.button("🔍 Predict Digit"):
        pred = model.predict(img_array)
        st.success(f"### ✅ Predicted Digit: **{pred[0]}**")
