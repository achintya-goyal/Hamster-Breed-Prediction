import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model("hamster_model")
print("✅ Model loaded")

# ==============================
# SETTINGS
# ==============================

img_size = (224, 224)
last_conv_layer_name = "Conv_1"  # MobileNetV2 last conv layer

class_names = ['dwarf', 'rats', 'syrian']

# ==============================
# GRAD MODEL (IMPORTANT)
# ==============================

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

test_dir = "../test_images"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    print(f"\nProcessing: {img_name}")

    # ==============================
    # LOAD IMAGE
    # ==============================
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ==============================
    # PREDICT
    # ==============================
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]

    print("Prediction:", pred_class)

    # ==============================
    # GRAD-CAM
    # ==============================
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # ==============================
    # SUPERIMPOSE
    # ==============================
    img_pil = Image.open(img_path).resize(img_size)
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = heatmap_resized * 0.4 + img_cv

    # ==============================
    # SHOW
    # ==============================
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM: {pred_class}")
    plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()