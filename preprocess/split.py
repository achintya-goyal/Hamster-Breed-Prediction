import os
import random
import shutil

# ==============================
# CONFIG
# ==============================

base_dir = "../Images"   # your original dataset
split_ratio = 0.15       # 15% validation

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# ==============================
# SPLIT DATA
# ==============================

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)

    # Skip non-folders + already split folders
    if not os.path.isdir(class_path) or class_name in ["train", "val"]:
        continue

    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    split_idx = int(split_ratio * len(images))

    val_images = images[:split_idx]
    train_images = images[split_idx:]

    # Create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move files
    for img in train_images:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    for img in val_images:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

    print(f"{class_name}: {len(train_images)} train | {len(val_images)} val")

print("✅ Dataset split completed successfully!")