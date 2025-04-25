import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import gdown
import os


model_url = 'https://drive.google.com/uc?id=1TKfDU7-ov_73lFogQLw-P1FUk82ahLqe'

model_path = './models/nigel_model.keras'
os.makedirs(os.path.dirname(model_path), exist_ok=True)



if not os.path.exists(model_path):
   
    gdown.download(id='1TKfDU7-ov_73lFogQLw-P1FUk82ahLqe', output=model_path, quiet=False)



if os.path.exists(model_path):
    print("Model downloaded successfully.")
else:
    print("Model not found. Please check the download link or path.")


try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")
    st.error("There was an error loading the model. Please check the file and path.")


class_names = [     
    'Benign keratosis-like lesions',            
    'Melanocytic nevi',          
    'Melanoma',                    
]


st.title("🧬 Skin Disease Detection")
st.write("Upload an image of a skin condition to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼 Uploaded Image', use_container_width=True)


    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 

    img_array = preprocess_input(img_array) 

    # Make prediction
    prediction = model.predict(img_array)
    pred_probs = prediction[0]
    pred_index = np.argmax(pred_probs)

    # Display results
    st.write(f"Predicted class index: {pred_index}")
    pred_class = class_names[pred_index]
    confidence = pred_probs[pred_index]

    st.write(f"🔍 **Prediction:** {pred_class}")
    st.write(f"📊 Confidence: {confidence:.2f}")
    st.write("Raw prediction:", prediction)
    st.write("Shape:", prediction.shape)

    st.write("📈 **Class Probabilities:**")
    prob_dict = {label: float(prob) for label, prob in zip(class_names , pred_probs)}
    st.json(prob_dict)
