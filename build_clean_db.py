import os
import pickle
import cv2
import numpy as np

sface = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', '')
print("SFace loaded")

# Format expected by simple_face_api.py:
# {'persons': {'12_aziz_jlassi': {'embeddings': [...], 'features': {...}}, ...}}
clean_db = {'persons': {}}

for folder in sorted(os.listdir('training_data')):
    folder_path = os.path.join('training_data', folder)
    if not os.path.isdir(folder_path):
        continue
    cropped_path = os.path.join(folder_path, 'cropped')
    if not os.path.exists(cropped_path):
        continue
    
    embeddings = []
    for img_file in os.listdir(cropped_path):
        # Skip down (had multiple faces detected in some cases)
        if 'down' in img_file.lower():
            continue
        img_path = os.path.join(cropped_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            aligned = cv2.resize(img, (112, 112))
            emb = sface.feature(aligned)
            embeddings.append(emb.flatten())
            print(f"  + {img_file}")
    
    if embeddings:
        clean_db['persons'][folder] = {
            'embeddings': np.array(embeddings),
            'features': {}
        }
        print(f"{folder}: {len(embeddings)} embeddings")

with open('recognition_database_clean.pkl', 'wb') as f:
    pickle.dump(clean_db, f)

print(f"\nSaved recognition_database_clean.pkl with {len(clean_db['persons'])} persons")
