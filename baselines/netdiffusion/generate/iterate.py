import os
import subprocess
import time
from tqdm import tqdm

# Constants
INPUT_IMAGE_DIR = "/home/rachel/ML-testing-dev/baselines/netdiffusion/fine_tune/kohya_ss_fork/model_training/test/image/15_network"
subscript = "test.py"

# Function to extract the numerical part of the filename
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

# Collect all image paths
input_images = [os.path.join(INPUT_IMAGE_DIR, img) for img in os.listdir(INPUT_IMAGE_DIR) if img.endswith('.png')]

# Sort the images based on the numerical value in the filename
input_images.sort(key=lambda x: extract_number(os.path.basename(x)))

# Start the timer
start_time = time.time()

# Iterate over each image and call the main script
for img_path in tqdm(input_images, desc="Processing Images", unit="image"):
    print(f"Processing image: {img_path}")
    subprocess.run(["python", subscript, img_path])

# End the timer
end_time = time.time()
total_time = end_time - start_time

# Print the total time taken
print(f"Total time taken: {total_time:.2f} seconds")
