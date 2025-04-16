import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load class names
class_names = tfds.builder('tf_flowers').info.features['label'].names

# Load the trained model
model = tf.keras.models.load_model('flower_model.h5')
print("✅ Model loaded!")

# Function to load and preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array, image

# Predict function
def classify_flower(image_path):
    image_array, original_image = preprocess_image(image_path)
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"🌸 Predicted Flower: {predicted_class} ({confidence*100:.2f}%)")

    # Display image with prediction
    plt.imshow(original_image)
    plt.title(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Replace 'your_flower.jpg' with your image file name
classify_flower("C:/Users/anifu/Downloads/Project_Predict_Flower/tulip1.jpeg")


