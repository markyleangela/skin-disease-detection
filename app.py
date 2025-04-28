import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# --- Download model if not already present ---
model_url = 'https://drive.google.com/uc?id=1T3BwgyvzZEDqZF_aH1muIEIJu4hlUSbc'
model_path = './models/better_model.keras'
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.exists(model_path):
    gdown.download(id='1T3BwgyvzZEDqZF_aH1muIEIJu4hlUSbc', output=model_path, quiet=False)

if os.path.exists(model_path):
    print("Model downloaded successfully.")
else:
    print("Model not found. Please check the download link or path.")


try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    st.error("There was an error loading the model. Please check the file and path.")


CLASS_NAMES = [
    'Benign keratosis-like lesions',
    'Melanocytic nevi',
    'Melanoma',
]


def preprocess_image(img_file):
    """Preprocess the uploaded image for prediction."""
    img = Image.open(img_file)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img, img_array

def predict(model, img_array):
    """Predict the class of the image."""
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class_index]
    predicted_class = CLASS_NAMES[pred_class_index]
    return predicted_class, confidence, preds[0]

# --- Streamlit App ---
st.title("🧬 Skin Disease Detection")
st.write("Upload an image of a skin condition to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Processing image...'):
        img, img_array = preprocess_image(uploaded_file)
        predicted_class, confidence, class_probs = predict(model, img_array)

    st.image(img, caption='🖼 Uploaded Image', use_container_width=True)

    st.success(f"🔍 **Prediction:** {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}")

    # (Optional) Show raw outputs
    with st.expander("See Raw Model Output"):
        st.write("Class probabilities array:", class_probs)

    # Show all class probabilities nicely
    st.write("📈 **Class Probabilities:**")
    prob_dict = {label: float(prob) for label, prob in zip(CLASS_NAMES, class_probs)}
    st.json(prob_dict)
