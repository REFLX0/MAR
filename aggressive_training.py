"""
Aggressive Training: Generate MANY more augmented images for better face discrimination.
This creates 50+ augmented images per person with diverse transformations.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import random

print("🚀 AGGRESSIVE TRAINING - Generating maximum augmentations...")
print("=" * 70)

training_path = Path('training_data')

def augment_image(img, augmentation_type):
    """Apply a specific augmentation to an image"""
    h, w = img.shape[:2]
    result = img.copy()
    
    if augmentation_type == 'brightness_up':
        result = cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(20, 50))
    elif augmentation_type == 'brightness_down':
        result = cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(-50, -20))
    elif augmentation_type == 'contrast_up':
        result = cv2.convertScaleAbs(img, alpha=random.uniform(1.2, 1.5), beta=0)
    elif augmentation_type == 'contrast_down':
        result = cv2.convertScaleAbs(img, alpha=random.uniform(0.6, 0.8), beta=0)
    elif augmentation_type == 'blur_light':
        result = cv2.GaussianBlur(img, (3, 3), 0)
    elif augmentation_type == 'blur_medium':
        result = cv2.GaussianBlur(img, (5, 5), 0)
    elif augmentation_type == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(img, -1, kernel)
    elif augmentation_type == 'flip_h':
        result = cv2.flip(img, 1)
    elif augmentation_type == 'rotate_slight_left':
        M = cv2.getRotationMatrix2D((w/2, h/2), random.randint(3, 8), 1.0)
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'rotate_slight_right':
        M = cv2.getRotationMatrix2D((w/2, h/2), random.randint(-8, -3), 1.0)
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'scale_up':
        scale = random.uniform(1.05, 1.15)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        result = scaled[y_start:y_start+h, x_start:x_start+w]
    elif augmentation_type == 'scale_down':
        scale = random.uniform(0.85, 0.95)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        result = np.zeros_like(img)
        x_start = (w - new_w) // 2
        y_start = (h - new_h) // 2
        result[y_start:y_start+new_h, x_start:x_start+new_w] = scaled
    elif augmentation_type == 'translate_left':
        M = np.float32([[1, 0, -random.randint(5, 15)], [0, 1, 0]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'translate_right':
        M = np.float32([[1, 0, random.randint(5, 15)], [0, 1, 0]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'translate_up':
        M = np.float32([[1, 0, 0], [0, 1, -random.randint(5, 15)]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'translate_down':
        M = np.float32([[1, 0, 0], [0, 1, random.randint(5, 15)]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif augmentation_type == 'noise_light':
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif augmentation_type == 'noise_medium':
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif augmentation_type == 'color_shift_warm':
        result = img.copy()
        result[:,:,2] = np.clip(result[:,:,2].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        result[:,:,0] = np.clip(result[:,:,0].astype(np.int16) - 10, 0, 255).astype(np.uint8)
    elif augmentation_type == 'color_shift_cool':
        result = img.copy()
        result[:,:,0] = np.clip(result[:,:,0].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        result[:,:,2] = np.clip(result[:,:,2].astype(np.int16) - 10, 0, 255).astype(np.uint8)
    elif augmentation_type == 'gamma_up':
        gamma = random.uniform(1.2, 1.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(img, table)
    elif augmentation_type == 'gamma_down':
        gamma = random.uniform(0.6, 0.8)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(img, table)
    elif augmentation_type == 'equalize':
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            result = cv2.equalizeHist(img)
    elif augmentation_type == 'clahe':
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(img)
    
    return result

# All augmentation types
AUGMENTATIONS = [
    'brightness_up', 'brightness_down',
    'contrast_up', 'contrast_down', 
    'blur_light', 'blur_medium', 'sharpen',
    'flip_h',
    'rotate_slight_left', 'rotate_slight_right',
    'scale_up', 'scale_down',
    'translate_left', 'translate_right', 'translate_up', 'translate_down',
    'noise_light', 'noise_medium',
    'color_shift_warm', 'color_shift_cool',
    'gamma_up', 'gamma_down',
    'equalize', 'clahe'
]

# Combined augmentations (apply 2 at once)
COMBINED_AUGS = [
    ('brightness_up', 'blur_light'),
    ('brightness_down', 'noise_light'),
    ('contrast_up', 'sharpen'),
    ('flip_h', 'brightness_up'),
    ('flip_h', 'brightness_down'),
    ('flip_h', 'contrast_up'),
    ('rotate_slight_left', 'brightness_up'),
    ('rotate_slight_right', 'brightness_down'),
    ('scale_up', 'sharpen'),
    ('gamma_up', 'noise_light'),
    ('gamma_down', 'blur_light'),
    ('clahe', 'sharpen'),
    ('equalize', 'noise_light'),
]

for folder in sorted(training_path.iterdir()):
    if not folder.is_dir():
        continue
    
    print(f"\n📁 Processing: {folder.name}")
    
    # Get source images from cropped folder
    cropped_path = folder / 'cropped'
    augmented_path = folder / 'augmented'
    
    if not cropped_path.exists():
        print("  ⚠️ No cropped folder, skipping")
        continue
    
    # Clear old augmented folder
    if augmented_path.exists():
        for f in augmented_path.glob('*.jpg'):
            f.unlink()
    else:
        augmented_path.mkdir(exist_ok=True)
    
    # Load source images (skip 'down' images)
    source_images = []
    for img_file in cropped_path.glob('*.jpg'):
        if 'down' in img_file.name.lower():
            continue
        img = cv2.imread(str(img_file))
        if img is not None:
            source_images.append((img_file.stem, img))
    
    print(f"  📷 Source images: {len(source_images)}")
    
    aug_count = 0
    
    # Generate single augmentations
    for src_name, src_img in source_images:
        for aug_type in AUGMENTATIONS:
            try:
                aug_img = augment_image(src_img, aug_type)
                aug_filename = f"{src_name}_{aug_type}.jpg"
                cv2.imwrite(str(augmented_path / aug_filename), aug_img)
                aug_count += 1
            except Exception as e:
                pass
    
    print(f"  ✅ Single augmentations: {aug_count}")
    
    # Generate combined augmentations
    combined_count = 0
    for src_name, src_img in source_images:
        for aug1, aug2 in COMBINED_AUGS:
            try:
                temp_img = augment_image(src_img, aug1)
                aug_img = augment_image(temp_img, aug2)
                aug_filename = f"{src_name}_{aug1}_{aug2}.jpg"
                cv2.imwrite(str(augmented_path / aug_filename), aug_img)
                combined_count += 1
            except Exception as e:
                pass
    
    print(f"  ✅ Combined augmentations: {combined_count}")
    
    # Generate random combination augmentations
    random_count = 0
    for src_name, src_img in source_images:
        for i in range(10):  # 10 random combinations per source image
            try:
                augs = random.sample(AUGMENTATIONS, 2)
                temp_img = augment_image(src_img, augs[0])
                aug_img = augment_image(temp_img, augs[1])
                aug_filename = f"{src_name}_random_{i}.jpg"
                cv2.imwrite(str(augmented_path / aug_filename), aug_img)
                random_count += 1
            except Exception as e:
                pass
    
    print(f"  ✅ Random augmentations: {random_count}")
    
    total = aug_count + combined_count + random_count
    print(f"  📊 TOTAL augmented images: {total}")

print("\n" + "=" * 70)
print("✅ Aggressive augmentation complete!")
print("\nNow rebuilding databases with more embeddings...")

# Now rebuild the enhanced database with ALL augmented images
print("\n" + "=" * 70)
print("Building ENHANCED recognition database with ALL augmentations...")

sface = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', '')
enhanced_db = {'persons': {}}

for folder in sorted(training_path.iterdir()):
    if not folder.is_dir():
        continue
    
    print(f"\n📁 Processing: {folder.name}")
    
    embeddings = []
    
    # 1. Add cropped images
    cropped_path = folder / 'cropped'
    if cropped_path.exists():
        for img_file in cropped_path.glob('*.jpg'):
            if 'down' in img_file.name.lower():
                continue
            img = cv2.imread(str(img_file))
            if img is not None:
                aligned = cv2.resize(img, (112, 112))
                emb = sface.feature(aligned).flatten()
                embeddings.append(emb)
        print(f"  + {len(embeddings)} from cropped/")
    
    # 2. Add ALL augmented images
    augmented_path = folder / 'augmented'
    aug_count = 0
    if augmented_path.exists():
        for img_file in augmented_path.glob('*.jpg'):
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
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-6)
        
        enhanced_db['persons'][folder.name] = {
            'embeddings': embeddings_array,
            'mean_embedding': mean_embedding,
            'count': len(embeddings)
        }
        print(f"  ✅ Total: {len(embeddings)} embeddings")

# Save enhanced database
import pickle
with open('recognition_database_enhanced.pkl', 'wb') as f:
    pickle.dump(enhanced_db, f)
print(f"\n✅ Saved recognition_database_enhanced.pkl")

# Now rebuild LBPH database with more images
print("\n" + "=" * 70)
print("Building LBPH database with ALL augmentations...")

LBPH_RADIUS = 2
LBPH_NEIGHBORS = 16
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

lbph_data = {
    'params': {
        'radius': LBPH_RADIUS,
        'neighbors': LBPH_NEIGHBORS,
        'grid_x': LBPH_GRID_X,
        'grid_y': LBPH_GRID_Y
    },
    'persons': {},
    'label_map': {}
}

current_label = 0

for folder in sorted(training_path.iterdir()):
    if not folder.is_dir():
        continue
    
    person_name = folder.name
    print(f"\n📁 Processing: {person_name}")
    
    person_images = []
    
    # 1. Add cropped images
    cropped_path = folder / 'cropped'
    if cropped_path.exists():
        for img_file in cropped_path.glob('*.jpg'):
            if 'down' in img_file.name.lower():
                continue
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                img_eq = cv2.equalizeHist(img_resized)
                person_images.append(img_eq)
        print(f"  + {len(person_images)} from cropped/")
    
    # 2. Add ALL augmented images
    augmented_path = folder / 'augmented'
    aug_count = 0
    if augmented_path.exists():
        for img_file in augmented_path.glob('*.jpg'):
            if 'down' in img_file.name.lower():
                continue
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                img_eq = cv2.equalizeHist(img_resized)
                person_images.append(img_eq)
                aug_count += 1
        print(f"  + {aug_count} from augmented/")
    
    if person_images:
        # Train LBPH for this person to extract histograms
        temp_lbph = cv2.face.LBPHFaceRecognizer_create(
            radius=LBPH_RADIUS,
            neighbors=LBPH_NEIGHBORS,
            grid_x=LBPH_GRID_X,
            grid_y=LBPH_GRID_Y
        )
        temp_labels = np.zeros(len(person_images), dtype=np.int32)
        temp_lbph.train(person_images, temp_labels)
        histograms = temp_lbph.getHistograms()
        
        lbph_data['persons'][person_name] = {
            'label': current_label,
            'histograms': [h.flatten() for h in histograms],
            'mean_histogram': np.mean([h.flatten() for h in histograms], axis=0)
        }
        lbph_data['label_map'][current_label] = person_name
        
        print(f"  ✅ Total: {len(person_images)} images, {len(histograms)} histograms")
        current_label += 1

# Save LBPH database
with open('lbph_database.pkl', 'wb') as f:
    pickle.dump(lbph_data, f)
print(f"\n✅ Saved lbph_database.pkl")

# Print final statistics
print("\n" + "=" * 70)
print("📊 FINAL TRAINING STATISTICS:")
print("=" * 70)

for person_name, data in enhanced_db['persons'].items():
    print(f"  {person_name}: {data['count']} SFace embeddings")

print("\n📊 Cross-person similarity (lower = better discrimination):")
persons = list(enhanced_db['persons'].keys())
for i in range(len(persons)):
    for j in range(i+1, len(persons)):
        emb_i = enhanced_db['persons'][persons[i]]['mean_embedding']
        emb_j = enhanced_db['persons'][persons[j]]['mean_embedding']
        cos_sim = np.dot(emb_i, emb_j)
        print(f"  {persons[i]} vs {persons[j]}: cosine={cos_sim:.3f}")

print("\n✅ AGGRESSIVE TRAINING COMPLETE!")
print("   Now test with: python test_recognition.py")
