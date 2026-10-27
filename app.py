
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

CLASSES = ["blackspot", "canker", "fresh", "grenning"]
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="AvocadoVision AI",
    page_icon="🥑",
    layout="centered",
)

st.title("🥑 AvocadoVision AI")
st.subheader("Avocado Ripening Stage Detection")
st.write(
    "Upload an avocado image to classify it using "
    "Classical Machine Learning or MobileNetV2."
)


def extract_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    for color_image in (hsv, lab):
        for channel in cv2.split(color_image):
            hist = cv2.calcHist(
                [channel], [0], None, [16], [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(12), range=(0, 11)
    )
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= max(lbp_hist.sum(), 1)

    features.extend(lbp_hist)
    return np.asarray(features, dtype=np.float32).reshape(1, -1)


@st.cache_resource
def load_random_forest():
    path = MODELS / "randomforest.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_mobilenet():
    path = MODELS / "mobilenetv2.pth"
    if not path.exists():
        return None

    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=True
    )

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASSES)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict_classical(image_rgb):
    model = load_random_forest()
    if model is None:
        st.error("RandomForest model missing. Train it first.")
        return None

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    features = extract_features(image_bgr)
    probabilities = model.predict_proba(features)[0]
    index = int(np.argmax(probabilities))

    return model.classes_[index], float(probabilities[index])


def predict_deep_learning(image_rgb):
    model = load_mobilenet()
    if model is None:
        st.error("MobileNetV2 model missing. Train it first.")
        return None

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    image = Image.fromarray(image_rgb)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(torch.argmax(probabilities).item())
    return CLASSES[index], float(probabilities[index].item())


uploaded_file = st.file_uploader(
    "Upload avocado image",
    type=["jpg", "jpeg", "png", "webp"],
)

model_choice = st.selectbox(
    "Choose model",
    ["RandomForest (Classical ML)", "MobileNetV2 (Deep Learning)"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    st.image(image, caption="Uploaded avocado", width=350)

    if st.button("Predict Ripening Stage", type="primary"):
        with st.spinner("Analyzing image..."):
            if model_choice.startswith("RandomForest"):
                result = predict_classical(image_rgb)
            else:
                result = predict_deep_learning(image_rgb)

        if result is not None:
            label, confidence = result
            st.success(f"Predicted class: {label.title()}")
            st.metric("Model confidence", f"{confidence * 100:.2f}%")
            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee of correct classification."
            )

            descriptions = {
                "blackspot": "Black spot symptoms detected.",
                "canker": "Canker symptoms detected.",
                "fresh": "Classified as fresh.",
                "grenning": "Classified as grenning.",
            }
            st.info(descriptions.get(label, "Prediction complete."))

st.divider()
st.caption(
    "AvocadoVision AI | Academic prototype. "
    "Predictions may be affected by lighting, image quality, "
    "and differences between training images and real-world images."
)
