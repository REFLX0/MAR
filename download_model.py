import urllib.request
import os

# Corrected URL (main branch)
url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
filename = "face_recognition_sface_2021dec.onnx"

print(f"Downloading {filename}...")
try:
    urllib.request.urlretrieve(url, filename)
    print("✅ Download complete!")
except Exception as e:
    print(f"❌ Download failed: {e}")
