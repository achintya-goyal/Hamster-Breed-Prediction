from PIL import Image
import os

base_dir = "../Images"

bad_files = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()  # check if valid
        except:
            print("Removing:", path)
            bad_files.append(path)

# delete bad files
for file in bad_files:
    try:
        os.remove(file)
    except:
        pass

print(f"✅ Removed {len(bad_files)} corrupted images")