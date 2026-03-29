import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ==============================
# LOAD FULL MODEL
# ==============================

model = tf.keras.models.load_model("hamster_model")
print("✅ Model loaded successfully")

# ==============================
# CLASS NAMES (IMPORTANT)
# ==============================

# ⚠️ MUST match training order
class_names = ['dwarf', 'rats', 'syrian']

# ==============================
# IMAGE SETTINGS
# ==============================

img_size = (224, 224)

# ==============================
# PREDICT FUNCTION
# ==============================

def predict_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Print result
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 30)

    except Exception as e:
        print(f"Skipping {img_path}: {e}")

# ==============================
# RUN ON TEST FOLDER
# ==============================

test_dir = "../test_images"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    predict_image(img_path)