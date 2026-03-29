import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path fix
base_dir = "../Images"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

img_size = (224, 224)
batch_size = 32

# 🔥 TRAIN AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.2,
    shear_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest'
)

# 🔵 VALIDATION (NO AUGMENTATION)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)

# 🧪 Show augmented samples
images, labels = next(train_data)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')

plt.show()