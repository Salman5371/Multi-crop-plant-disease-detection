import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import build_efficientnetv2s
from src.decision_support import (
    estimate_preliminary_severity,
    get_management_recommendation,
    generate_alert
)


# =========================
# App configuration
# =========================

st.set_page_config(
    page_title="Smart Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# =========================
# Paths and settings
# =========================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATH = ROOT_DIR / "models" / "efficientnetv2s_best.pth"
CLASS_NAMES_PATH = ROOT_DIR / "Data" / "class_names.json"

OUTPUT_DIR = ROOT_DIR / "outputs" / "prediction_reports"
TEMP_DIR = ROOT_DIR / "outputs" / "temp_uploads"

IMAGE_SIZE = 224
DISPLAY_IMAGE_SIZE = 450
TOP_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helper functions
# =========================

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path, num_classes):
    model = build_efficientnetv2s(num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = load_class_names(CLASS_NAMES_PATH)
    model = load_model(MODEL_PATH, num_classes=len(class_names))

    return model, class_names


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def create_display_image(image, max_size=DISPLAY_IMAGE_SIZE):
    """
    Resize image only for Streamlit display.
    This does not affect model prediction.
    """
    display_image = image.copy()
    display_image.thumbnail((max_size, max_size))
    return display_image


def extract_crop_and_disease(class_name):
    cleaned = class_name.replace("__", "_")
    parts = cleaned.split("_")

    crop = parts[0].title()

    if len(parts) > 1:
        disease = " ".join(parts[1:]).title()
    else:
        disease = "Unknown"

    return crop, disease


def predict_image(model, image, class_names, top_k=TOP_K):
    """
    Predict Top-1 and Top-K classes.
    """
    image_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    top_predictions = []

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        class_name = class_names[int(idx)]
        crop, disease = extract_crop_and_disease(class_name)

        top_predictions.append({
            "Rank": rank,
            "Predicted Class": class_name,
            "Crop": crop,
            "Disease / Status": disease,
            "Confidence (%)": round(float(prob) * 100, 2)
        })

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(top_probs[0])

    return predicted_class, confidence, predicted_index, top_predictions


def save_uploaded_image(uploaded_file):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    save_path = TEMP_DIR / f"uploaded_{timestamp}{file_extension}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


def save_prediction_report(
    image_name,
    predicted_class,
    predicted_index,
    crop,
    disease,
    confidence,
    top_predictions,
    severity,
    severity_note,
    recommendations,
    alert
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"app_prediction_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SMART FARMING DISEASE DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input Information\n")
        f.write("-" * 60 + "\n")
        f.write(f"Image Name       : {image_name}\n\n")

        f.write("Top-1 Prediction Result\n")
        f.write("-" * 60 + "\n")
        f.write(f"Predicted Index  : {predicted_index}\n")
        f.write(f"Predicted Class  : {predicted_class}\n")
        f.write(f"Crop             : {crop}\n")
        f.write(f"Disease/Status   : {disease}\n")
        f.write(f"Confidence       : {confidence * 100:.2f}%\n\n")

        f.write("Top-5 Prediction Results\n")
        f.write("-" * 60 + "\n")
        for item in top_predictions:
            f.write(
                f"Rank {item['Rank']}: "
                f"{item['Predicted Class']} | "
                f"Crop: {item['Crop']} | "
                f"Disease/Status: {item['Disease / Status']} | "
                f"Confidence: {item['Confidence (%)']}%\n"
            )

        f.write("\nSeverity / Risk Indication\n")
        f.write("-" * 60 + "\n")
        f.write(f"Level            : {severity}\n")
        f.write(f"Note             : {severity_note}\n\n")

        f.write("Management Recommendation\n")
        f.write("-" * 60 + "\n")
        for i, rec in enumerate(recommendations, start=1):
            f.write(f"{i}. {rec}\n")

        f.write("\nAlert Message\n")
        f.write("-" * 60 + "\n")
        f.write(alert)

    return report_path


# =========================
# Streamlit UI
# =========================

st.title("🌿 Smart Crop Disease Detection and Management System")

st.markdown(
    """
This web application uses a trained **EfficientNetV2-S** deep learning model to detect crop diseases from leaf images.

It provides:
- crop and disease prediction
- Top-5 prediction results
- confidence score
- preliminary severity/risk indication
- rule-based management recommendation
- farmer alert message
- downloadable prediction report
"""
)

st.sidebar.header("System Information")
st.sidebar.write(f"**Device:** {DEVICE}")
st.sidebar.write("**Model:** EfficientNetV2-S")
st.sidebar.write("**Input size:** 224 × 224")
st.sidebar.write(f"**Display image size:** Max {DISPLAY_IMAGE_SIZE} × {DISPLAY_IMAGE_SIZE}")
st.sidebar.write(f"**Top-K predictions:** {TOP_K}")

try:
    model, class_names = load_model_and_classes()
    st.sidebar.success("Model loaded successfully")
    st.sidebar.write(f"**Classes:** {len(class_names)}")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    saved_image_path = save_uploaded_image(uploaded_file)

    st.subheader("Uploaded Leaf Image")

    display_image = create_display_image(image)
    st.image(display_image, caption=uploaded_file.name)

    st.caption(
        "Note: For better prediction, upload a clear close-up leaf image without watermark, text, heavy background, or blur."
    )

    with st.spinner("Analyzing image..."):
        predicted_class, confidence, predicted_index, top_predictions = predict_image(
            model=model,
            image=image,
            class_names=class_names,
            top_k=TOP_K
        )

        crop, disease = extract_crop_and_disease(predicted_class)

        severity, severity_note = estimate_preliminary_severity(
            predicted_class=predicted_class,
            confidence=confidence
        )

        recommendations = get_management_recommendation(predicted_class)

        alert = generate_alert(
            crop=crop,
            disease=disease,
            confidence=confidence,
            severity=severity
        )

        report_path = save_prediction_report(
            image_name=uploaded_file.name,
            predicted_class=predicted_class,
            predicted_index=predicted_index,
            crop=crop,
            disease=disease,
            confidence=confidence,
            top_predictions=top_predictions,
            severity=severity,
            severity_note=severity_note,
            recommendations=recommendations,
            alert=alert
        )

    st.success("Prediction completed")

    st.subheader("Top-1 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Crop", crop)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    with col2:
        st.metric("Disease / Status", disease)
        st.metric("Predicted Index", predicted_index)

    st.write(f"**Predicted Class:** `{predicted_class}`")

    if confidence < 0.70:
        st.warning(
            "The model confidence is low. Please upload another clear image or verify manually."
        )

    if confidence >= 0.90:
        st.info(
            "The model confidence is high. However, high confidence does not always guarantee correctness, especially for noisy or out-of-distribution images."
        )

    st.subheader("Top-5 Prediction Results")

    top5_df = pd.DataFrame(top_predictions)
    st.dataframe(top5_df, use_container_width=True)

    st.subheader("Severity / Risk Indication")
    st.warning(f"**Level:** {severity}")
    st.write(f"**Note:** {severity_note}")

    st.subheader("Management Recommendation")
    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")

    st.subheader("Farmer Alert Message")
    st.info(alert)

    st.subheader("Saved Report")
    st.write("Prediction report saved at:")
    st.code(str(report_path))

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    st.download_button(
        label="Download Prediction Report",
        data=report_text,
        file_name=report_path.name,
        mime="text/plain"
    )

else:
    st.info("Please upload a plant leaf image to start prediction.")