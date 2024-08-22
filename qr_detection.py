from pyzbar import pyzbar
import cv2
import os
from glob import glob
import shutil
from PIL import Image 

# Directory containing images
image_input_path = "qrDetection/input"
image_output_path = "qrDetection/output"
print(image_input_path)

def generate_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename) and counter<3):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# Process each image in the directory
image_names = glob('*.jpg', root_dir=image_input_path)

for file_name in image_names:
    file_path = os.path.join(image_input_path, file_name) 
    print(f"Processing file: {file_name}")

    # Load and preprocess the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {file_name}")
        continue

    # Find the barcodes in the preprocessed image
    barcodes = pyzbar.decode(image)
    
    if not barcodes:
        print(f"No barcodes found in {file_name}.")
        
    else:
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            print(barcode_data)

            # Sanitize the barcode data to create a valid filename
            filename = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in barcode_data)

            # Construct new file path
            new_file_name = f"{filename}.jpg"
            new_file_path = os.path.join(image_input_path, new_file_name)

            # Generate a unique filename if necessary
            unique_file_name = generate_unique_filename(image_input_path, new_file_name)
            unique_file_path = os.path.join(image_input_path, unique_file_name)

            # Rename the file
            try:
                os.rename(file_path, unique_file_path)
                print(f"File {file_name} renamed to: {unique_file_name}")
                shutil.move(unique_file_path, image_output_path)
            except Exception as e:
                print(f"Failed to rename file {file_name} to {unique_file_name}: {e}")

            # Break because one barcode per image is expected
            break
