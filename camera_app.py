# digit_recognition_offline_sklearn.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train a simple model (offline, no TensorFlow)
@st.cache_resource
def train_model():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

model, acc = train_model()

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✏️")
st.title("📷 Handwritten Digit Recognition (Offline, No TensorFlow)")
st.write(f"Model trained with scikit-learn. Accuracy: **{acc*100:.2f}%**")

file = st.camera_input("Take a photo of your digit")

if file is not None:
    img = Image.open(file).convert("L")  # grayscale
    img = img.resize((8, 8), Image.LANCZOS)  # match sklearn digits size
    img = ImageOps.invert(img)  # digits dataset has white digits on black
    img_array = np.array(img) / 16.0  # scale to 0-16 range
    img_array = img_array.reshape(1, -1)

    st.image(img, caption="Processed Image (8x8)", width=150)

    if st.button("Predict"):
        pred = model.predict(img_array)[0]
        st.subheader(f"Predicted Digit: **{pred}**")
