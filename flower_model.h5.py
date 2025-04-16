import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import streamlit as st

# Step 1: Load Dataset
print("🔄 Loading dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print("✅ Dataset loaded!")

# Step 2: Preprocess
IMG_SIZE = 224

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array



ds_train = ds_train.map(format_example).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(format_example).batch(32).prefetch(tf.data.AUTOTUNE)

# Step 3: Load Pretrained Base Model
print("🔧 Building model...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train the Model
print("🚀 Starting training...")
history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=5,
    verbose=1  # Show progress bar
)
print("✅ Training complete!")

# Step 5: Evaluate
loss, acc = model.evaluate(ds_test, verbose=1)
print(f"📊 Final Test Accuracy: {acc * 100:.2f}%")

# Step 6: Save the Model
model.save('flower_model.h5')
print("💾 Model saved as 'flower_model.h5'")

# Step 7: Show Example Predictions
print("\n🔍 Showing example predictions...")
class_names = ds_info.features['label'].names

for images, labels in ds_test.take(1):
    preds = model.predict(images)
    for i in range(5):
        plt.imshow(images[i])
        plt.title(f"Predicted: {class_names[tf.argmax(preds[i])]} | True: {class_names[labels[i]]}")
        plt.axis('off')
        plt.show()
