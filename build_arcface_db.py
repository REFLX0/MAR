"""
Build ArcFace recognition database from training images.
Uses InsightFace buffalo_l model for better accuracy.
"""
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from insightface.app import FaceAnalysis

print("Building ArcFace recognition database...")
print("=" * 60)

# Initialize InsightFace
print("Loading ArcFace model...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ ArcFace model loaded!")

training_path = Path('training_data')
db_data = {'persons': {}, 'model': 'arcface-buffalo_l'}

for folder in sorted(training_path.iterdir()):
    if not folder.is_dir():
        continue
    
    person_name = folder.name
    print(f"\n📁 Processing: {person_name}")
    
    embeddings = []
    
    # 1. Process cropped images (original photos)
    cropped_path = folder / 'cropped'
    if cropped_path.exists():
        cropped_count = 0
        for img_file in sorted(cropped_path.glob('*.jpg')):
            if 'down' in img_file.name.lower():
                continue
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Get face embedding
            faces = face_app.get(img)
            if faces:
                # Use largest face
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                embeddings.append(face.embedding)
                cropped_count += 1
        
        print(f"  + {cropped_count} from cropped/")
    
    # 2. Process augmented images (for variety)
    augmented_path = folder / 'augmented'
    if augmented_path.exists():
        aug_count = 0
        # Use all augmented images for better coverage
        for img_file in augmented_path.glob('*.jpg'):
            if 'down' in img_file.name.lower():
                continue
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            faces = face_app.get(img)
            if faces:
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                embeddings.append(face.embedding)
                aug_count += 1
        
        print(f"  + {aug_count} from augmented/")
    
    if embeddings:
        emb_array = np.array(embeddings)
        mean_emb = np.mean(emb_array, axis=0)
        
        db_data['persons'][person_name] = {
            'embeddings': emb_array,
            'mean_embedding': mean_emb
        }
        
        print(f"  ✅ Total: {len(embeddings)} embeddings")

# Save database
output_path = Path('recognition_database_arcface.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(db_data, f)
print(f"\n✅ Saved: {output_path}")

# Analyze similarity between persons
print("\n" + "=" * 60)
print("Analyzing ArcFace embedding similarities...")
print()

persons = list(db_data['persons'].keys())
print("Cosine Similarity between persons (higher = more similar):")
for i in range(len(persons)):
    for j in range(i+1, len(persons)):
        emb_i = db_data['persons'][persons[i]]['mean_embedding']
        emb_j = db_data['persons'][persons[j]]['mean_embedding']
        
        # Normalize
        emb_i_norm = emb_i / (np.linalg.norm(emb_i) + 1e-6)
        emb_j_norm = emb_j / (np.linalg.norm(emb_j) + 1e-6)
        
        cos_sim = np.dot(emb_i_norm, emb_j_norm)
        l2_dist = np.linalg.norm(emb_i - emb_j)
        
        print(f"  {persons[i][:20]:20} vs {persons[j][:20]:20}: cos={cos_sim:.4f}, L2={l2_dist:.3f}")

print("\n✅ ArcFace database built successfully!")
