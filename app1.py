import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

model = load_model()

st.title("Handwritten Digit Recognition with Background Black and Digit White")

def preprocess_image(img):
    # Convert to grayscale (if not already)
    img_gray = img.convert('L') if img.mode != 'L' else img
    img_array = np.array(img_gray)
    
    # Threshold to binary
    _, binary_img = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    
    # Invert colors: make background black, digit white
    inverted_img = cv2.bitwise_not(binary_img)
    
    # Resize to model input size 28x28
    resized_img = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0,1]
    norm_img = resized_img / 255.0
    return norm_img

def predict_digit(img_array):
    input_img = img_array.reshape(1, 28, 28, 1).astype('float32')
    prediction = model.predict(input_img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_digit, confidence

# ---------------------------
# Input Selection
# ---------------------------
st.subheader("Choose an input method:")
method = st.radio("📥 Input Method", ["Camera", "Upload File"], horizontal=True)

image = None
if method == "Camera":
    img_file_buffer = st.camera_input("Capture your handwritten digit")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
elif method == "Upload File":
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# ---------------------------
# Process and Predict
# ---------------------------
if image is not None:
    processed_img = preprocess_image(image)
    
    # Display processed image (background black, digit white)
    st.image(processed_img, caption="Processed Image (Background=Black, Digit=White)", width=150, clamp=True)
    
    # Predict
    digit, conf = predict_digit(processed_img)
    st.markdown(f"### Predicted Digit: {digit}")
    st.markdown(f"Confidence: {conf:.2f}")

# Tips
st.markdown("""
#### Tips:
- Write the digit clearly with a dark pen on white paper.
- Hold it steady and centered while capturing or upload a clear image.
""")
