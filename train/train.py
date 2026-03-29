import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# CHECK GPU
# ==============================

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ==============================
# PATHS
# ==============================

base_dir = "../Images"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# ==============================
# IMAGE SETTINGS
# ==============================

img_size = (224, 224)
batch_size = 32

# ==============================
# DATA GENERATORS (AUGMENTATION)
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.2,
    shear_range=0.1,

    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
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

num_classes = train_data.num_classes
print("Classes:", train_data.class_indices)

# ==============================
# LOAD MODEL (EfficientNet)
# ==============================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# ==============================
# FREEZE MOST LAYERS
# ==============================

base_model.trainable = False

# Freeze early layers, train last few
# for layer in base_model.layers[:-50]:
#     layer.trainable = False

# ==============================
# CUSTOM CLASSIFIER (HEAD)
# ==============================

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)

output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# EARLY STOPPING
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==============================
# TRAIN MODEL
# ==============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

# ==============================
# SAVE MODEL
# ==============================

model.save("hamster_model", save_format="tf")

print("✅ Training complete!")