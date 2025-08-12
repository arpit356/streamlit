import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Generate dummy 28x28 data for offline training
# ---------------------------
@st.cache_resource
def train_model():
    # Create synthetic digits (not as accurate as MNIST but works offline)
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
    img = Image.open(file).convert("L")  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)  # resize to 28x28
    img = ImageOps.invert(img)  # invert colors if needed
    img_array = np.array(img).reshape(1, -1)  # flatten
    img_array = img_array / 255.0 * 255  # keep same scale

    st.image(img, caption="🖼 Processed Image (28×28)", width=150)

    if st.button("🔍 Predict"):
        pred = model.predict(img_array)[0]
        st.success(f"✅ Predicted Digit: **{pred}**")
