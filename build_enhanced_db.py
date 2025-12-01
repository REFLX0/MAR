"""
Build enhanced recognition database with ALL augmented images for better discrimination.
Uses both cosine similarity and L2 distance for matching.
"""
import os
import pickle
import cv2
import numpy as np
from pathlib import Path

print("Building ENHANCED recognition database...")
print("=" * 60)

sface = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', '')
print("✅ SFace loaded")

# Format: {'persons': {'12_aziz_jlassi': {'embeddings': [...], 'mean_embedding': [...]}}}
enhanced_db = {'persons': {}}

training_path = Path('training_data')

for folder in sorted(training_path.iterdir()):
    if not folder.is_dir():
        continue
    
    print(f"\n📁 Processing: {folder.name}")
    
    embeddings = []
    
    # 1. Add cropped images (clean, high quality)
    cropped_path = folder / 'cropped'
    if cropped_path.exists():
        for img_file in cropped_path.glob('*.jpg'):
            # Skip 'down' images (often have detection issues)
            if 'down' in img_file.name.lower():
                continue
            img = cv2.imread(str(img_file))
            if img is not None:
                aligned = cv2.resize(img, (112, 112))
                emb = sface.feature(aligned).flatten()
                embeddings.append(emb)
        print(f"  + {len(embeddings)} from cropped/")
    
    # 2. Add ALL augmented images for more variety
    augmented_path = folder / 'augmented'
    aug_count = 0
    if augmented_path.exists():
        for img_file in augmented_path.glob('*.jpg'):  # Use ALL augmentations
            # Skip down-based augmentations
            if 'down' in img_file.name.lower():
                continue
            img = cv2.imread(str(img_file))
            if img is not None:
                aligned = cv2.resize(img, (112, 112))
                emb = sface.feature(aligned).flatten()
                embeddings.append(emb)
                aug_count += 1
        print(f"  + {aug_count} from augmented/")
    
    if embeddings:
        embeddings_array = np.array(embeddings)
        
        # Calculate mean embedding for this person (centroid)
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-6)  # Normalize
        
        # Calculate std for quality check
        std_embedding = np.std(embeddings_array, axis=0)
        consistency = 1.0 / (np.mean(std_embedding) + 1e-6)
        
        enhanced_db['persons'][folder.name] = {
            'embeddings': embeddings_array,
            'mean_embedding': mean_embedding,
            'consistency': float(consistency),
            'count': len(embeddings)
        }
        print(f"  ✅ Total: {len(embeddings)} embeddings, consistency: {consistency:.2f}")

# Save
with open('recognition_database_enhanced.pkl', 'wb') as f:
    pickle.dump(enhanced_db, f)

print("\n" + "=" * 60)
print(f"✅ Saved recognition_database_enhanced.pkl")
print(f"   Persons: {len(enhanced_db['persons'])}")

# Verify discrimination between persons
print("\n📊 Cross-person similarity check (using MEAN embeddings):")
persons = list(enhanced_db['persons'].keys())
for i in range(len(persons)):
    for j in range(i+1, len(persons)):
        emb_i = enhanced_db['persons'][persons[i]]['mean_embedding']
        emb_j = enhanced_db['persons'][persons[j]]['mean_embedding']
        
        # Cosine similarity
        cos_sim = np.dot(emb_i, emb_j)
        
        # L2 distance (lower = more different)
        l2_dist = np.linalg.norm(emb_i - emb_j)
        
        name_i = persons[i].split('_', 1)[1] if '_' in persons[i] else persons[i]
        name_j = persons[j].split('_', 1)[1] if '_' in persons[j] else persons[j]
        
        status = "⚠️ CLOSE" if cos_sim > 0.6 else "✅ OK"
        print(f"  {name_i} vs {name_j}: cos={cos_sim:.3f}, L2={l2_dist:.3f} {status}")
