import os
import shutil

# Direktori sumber file JPG
jpg_source_dir = "train_images/train_images"

# Direktori file TIFF yang sesuai
tiff_dir = "output_tiff"

# Direktori tujuan untuk file JPG yang sesuai
destination_dir = "output_jpg"

# Buat direktori tujuan jika belum ada
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop melalui file TIFF di direktori "ouput_tiff"
for tiff_file in os.listdir(tiff_dir):
    # Pastikan bahwa file adalah file TIFF
    if tiff_file.endswith(".tiff"):
        # Mencari nama file JPG yang sesuai dengan format nama file TIFF
        jpg_file = tiff_file.replace(".tiff", ".jpg")

        # Mengecek apakah file JPG ada di direktori sumber
        if os.path.exists(os.path.join(jpg_source_dir, jpg_file)):
            # Buat path lengkap ke file JPG dan tujuan
            jpg_source_path = os.path.join(jpg_source_dir, jpg_file)
            jpg_destination_path = os.path.join(destination_dir, jpg_file)

            # Pindahkan file JPG yang sesuai ke direktori tujuan
            shutil.copy(jpg_source_path, jpg_destination_path)

print("Proses pemindahan selesai.")
