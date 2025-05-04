import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os


st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #4A90E2;
    }
    .sub-info {
        font-size: 1.1em;
        color: #444;
    }
    .prediction {
        background-color: #444;
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        font-size: 1.2em;
    }
    .confidence {
        font-size: 1em;
        color: #2e7d32;
        margin-top: -10px;
    }
    .footer {
        margin-top: 50px;
        font-size: 0.9em;
        color: #999;
    }
    </style>
""", unsafe_allow_html=True)

# --- Download model if not already present ---
model_url = 'https://drive.google.com/uc?id=1qtDZb4hyKbge8tE5E2TgmdOfK8B1IAit'
model_path = './models/better_model.keras'
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.exists(model_path):
    gdown.download(id='1qtDZb4hyKbge8tE5E2TgmdOfK8B1IAit', output=model_path, quiet=False)

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

CLASS_NAMES = [
    'Benign keratosis-like lesions',
    'Melanocytic nevi',
    'Melanoma',
]

CLASS_DETAILS = {
    "Benign keratosis-like lesions": (
        "🔵 **Benign keratosis-like lesions** are non-cancerous skin growths that include seborrheic keratoses, "
        "solar lentigines (age spots), and lichen planus-like keratoses. These lesions often appear as rough, "
        "scaly, or wart-like patches that can vary in color from brown to black. While they may resemble "
        "malignant lesions like melanoma, they are usually harmless. However, due to their visual similarity "
        "to more serious skin conditions, monitoring and occasional biopsy may be recommended to rule out malignancy."
    ),
    
    "Melanocytic nevi": (
        "🟠 **Melanocytic nevi** are commonly referred to as moles. They are made up of melanocytes, which are pigment-producing "
        "cells in the skin. These nevi are generally harmless and can appear anywhere on the body. They are usually brown, "
        "black, or flesh-colored, and can be flat or raised. Most people have 10–40 moles by adulthood. While the vast "
        "majority remain benign, changes in size, shape, or color—especially asymmetry or irregular borders—can be warning "
        "signs of malignant transformation and should be evaluated by a dermatologist."
    ),
    
    "Melanoma": (
        "🔴 **Melanoma** is a highly aggressive and potentially life-threatening form of skin cancer that develops from melanocytes. "
        "It can spread (metastasize) rapidly to other parts of the body if not detected and treated early. Melanomas often start "
        "as new or changing moles, especially those that are asymmetrical, have uneven borders, contain multiple colors, or grow "
        "in size. Early diagnosis through skin exams and biopsies greatly increases survival rates. Treatment may involve surgical "
        "removal, immunotherapy, radiation, or chemotherapy depending on the stage of the cancer."
    )
}


def preprocess_image(img_file):
    img = Image.open(img_file)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img, img_array

def predict(model, img_array):
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class_index]
    predicted_class = CLASS_NAMES[pred_class_index]
    return predicted_class, confidence, preds[0]


st.markdown('<div class="main-title">🧬 Skin Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-info">Upload an image of a skin condition to get a prediction using a trained AI model.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('🔄 Processing image...'):
        img, img_array = preprocess_image(uploaded_file)
        predicted_class, confidence, class_probs = predict(model, img_array)

    st.image(img, caption='🖼 Uploaded Image', use_container_width=True)

    st.markdown(f'<div class="prediction">🔍 <strong>Prediction:</strong> {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="confidence">📊 Confidence: {confidence:.2f}</p>', unsafe_allow_html=True)

    with st.expander("🔎 See Raw Model Output"):
        st.write("Class probabilities array:", class_probs)

    st.markdown("### ℹ️ Class Information")
    st.info(CLASS_DETAILS.get(predicted_class, "No additional details available."))


