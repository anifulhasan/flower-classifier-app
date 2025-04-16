import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

import os
print("Files in current directory:", os.listdir())

# Load your model and class names
model = load_model(r"C:\Users\anifu\Downloads\Project_Predict_Flower\flower_model.h5")
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # adjust if needed

st.set_page_config(page_title="Flower Classifier", layout="centered")
st.title("🌸 Flower Image Classifier")
st.write("Upload an image of a flower, and I’ll try to guess what it is!")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("Classifying...")

    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"🌼 I think this is a **{predicted_class}**!")
