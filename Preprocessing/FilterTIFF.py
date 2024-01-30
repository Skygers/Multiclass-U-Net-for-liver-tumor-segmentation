import cv2
import numpy as np
import os

# Membuat direktori output jika belum ada
output_dir = "output_tiff"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mendefinisikan fungsi untuk memeriksa np.unique pada gambar TIFF
def has_desired_unique_values(image_path, desired_unique_values):
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    unique_values = np.unique(image)
    return set(desired_unique_values).issubset(unique_values)

# Direktori dengan file TIFF
tiff_directory = "512x512_train_masks/train_masks/train_masks/"

# Membuat daftar file yang memiliki np.unique [0, 1, 2]
desired_unique_values = [0, 1, 2]
selected_files = []

for root, dirs, files in os.walk(tiff_directory):
    for file in files:
        if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
            tiff_path = os.path.join(root, file)
            if has_desired_unique_values(tiff_path, desired_unique_values):
                print(f"{tiff_path} telah disimpan")
                selected_files.append(tiff_path)

# Menyimpan file yang memenuhi kriteria
for tiff_path in selected_files:
    filename = os.path.basename(tiff_path)
    output_path = os.path.join(output_dir, filename)
    os.rename(tiff_path, output_path)

print(f"{len(selected_files)} file TIFF yang memenuhi kriteria telah disimpan di {output_dir}")
