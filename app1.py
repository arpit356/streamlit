import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Generate dummy 28x28 data for offline training
# ---------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    X = np.random.randint(0, 256, (2000, 28*28))
    y = np.random.randint(0, 10, 2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

model, acc = train_model()

# ---------------------------
# Page UI
# ---------------------------
st.set_page_config(page_title="Handwritten Digit Recognition 28x28", page_icon="✏️")
st.title("📷 Handwritten Digit Recognition (28×28, Offline)")
st.write(f"Model trained with RandomForest. Accuracy (synthetic data): **{acc*100:.2f}%**")

# ---------------------------
# Input Selection
# ---------------------------
st.subheader("Choose an input method:")
method = st.radio("📥 Input Method", ["Camera", "Upload File"], horizontal=True)

file = None
if method == "Camera":
    file = st.camera_input("Take a photo of your digit")
elif method == "Upload File":
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# ---------------------------
# Process Image & Predict
# ---------------------------
if file is not None:
    # Open image
    img_pil = Image.open(file).convert("L")  # grayscale
    img_np = np.array(img_pil)

    # ----- Segmentation -----
    # Blur + Threshold
    blur = cv2.GaussianBlur(img_np, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour (digit)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop to bounding box
        digit_crop = img_np[y:y+h, x:x+w]
    else:
        digit_crop = img_np  # fallback if no contour found

    # Resize to 28x28
    digit_pil = Image.fromarray(digit_crop)
    digit_pil = ImageOps.invert(digit_pil)  # white digit on black
    digit_pil = digit_pil.resize((28, 28), Image.LANCZOS)

    # Prepare for model
    img_array = np.array(digit_pil).reshape(1, -1)
    img_array = img_array / 255.0 * 255

    # Show processed
    st.image(digit_pil, caption="🖼 Segmented Digit (28×28)", width=150)

    # Predict
    if st.button("🔍 Predict"):
        pred = model.predict(img_array)[0]
        st.success(f"✅ Predicted Digit: **{pred}**")
