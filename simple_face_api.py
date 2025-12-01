import os
import cv2
import numpy as np
import base64
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
from PIL import Image
import glob
import time

# Import YOLOv11 detector
try:
    from face_detector_yolo import YOLOFaceDetector
    USE_YOLO = True
    print("✅ YOLOv11 detector loaded")
except Exception as e:
    USE_YOLO = False
    print(f"⚠️ YOLOv11 not available: {e}")

# Import feature extractor
try:
    from advanced_face_features import AdvancedFaceFeatures
    feature_extractor = AdvancedFaceFeatures()
    USE_FEATURES = True
    print("✅ Advanced face features loaded")
except Exception as e:
    USE_FEATURES = False
    feature_extractor = None
    print(f"⚠️ Advanced features not available: {e}")

# Pre-computed recognition database path
RECOGNITION_DB_PATH = Path('recognition_database_enhanced.pkl')  # Use enhanced DB

app = Flask(__name__)
CORS(app)

# Paths
TRAINING_DATA_PATH = Path('training_data')
TRAINING_DATA_PATH.mkdir(exist_ok=True)
FACE_DATABASE_PATH = Path('face_database')
FACE_DATABASE_PATH.mkdir(exist_ok=True)
MODEL_PATH = "face_recognition_sface_2021dec.onnx"

# Initialize Face Recognizer (SFace)
recognizer = cv2.FaceRecognizerSF.create(
    model=MODEL_PATH,
    config="",
    backend_id=0,
    target_id=0
)
print("✅ SFace Recognizer initialized")

# Initialize LBPH database (pickle format - faster loading)
USE_LBPH = False
lbph_data = None
LBPH_DB_PATH = Path('lbph_database.pkl')

if LBPH_DB_PATH.exists():
    try:
        print("🔄 Loading LBPH database...")
        with open(LBPH_DB_PATH, 'rb') as f:
            lbph_data = pickle.load(f)
        USE_LBPH = True
        print(f"✅ LBPH loaded: {len(lbph_data['persons'])} persons")
    except Exception as e:
        print(f"⚠️ LBPH load failed: {e}")
        USE_LBPH = False
else:
    print("⚠️ LBPH database not found - run build_lbph_db.py first")

# Initialize YOLOv11
if USE_YOLO:
    yolo_detector = YOLOFaceDetector(confidence_threshold=0.3)
    print("✅ YOLOv11 face detector initialized")

# In-memory database for fast matching
face_db = []
face_db_means = {}  # Mean embeddings per person for L2 matching

def load_database():
    """Load all faces from enhanced database with mean embeddings"""
    global face_db, face_db_means
    face_db = []
    face_db_means = {}
    print("🔄 Loading face database...")
    
    # Load enhanced database with mean embeddings
    if RECOGNITION_DB_PATH.exists():
        try:
            with open(RECOGNITION_DB_PATH, 'rb') as f:
                db_data = pickle.load(f)
            
            for person_key, person_data in db_data['persons'].items():
                parts = person_key.split('_', 1)
                if len(parts) < 2:
                    continue
                member_id = int(parts[0])
                name = parts[1].replace('_', ' ')
                
                # Store mean embedding for fast matching
                if 'mean_embedding' in person_data:
                    face_db_means[person_key] = {
                        'mean_embedding': person_data['mean_embedding'],
                        'id': member_id,
                        'name': name
                    }
                
                # Store individual embeddings for voting
                embeddings = person_data['embeddings']
                for emb in embeddings:
                    face_db.append({
                        'id': member_id,
                        'name': name,
                        'feature': emb.reshape(1, -1).astype(np.float32)
                    })
            
            print(f"✅ Loaded enhanced database: {len(face_db)} embeddings, {len(face_db_means)} persons")
            return
        except Exception as e:
            print(f"⚠️ Failed to load enhanced DB: {e}")
    
    # 2. Try loading from embeddings JSON files (per-person)
    for person_dir in TRAINING_DATA_PATH.glob('*'):
        if person_dir.is_dir():
            try:
                parts = person_dir.name.split('_', 1)
                if len(parts) < 2: continue
                member_id = int(parts[0])
                name = parts[1].replace('_', ' ')
                
                # Check for pre-computed embeddings
                emb_file = person_dir / 'embeddings' / 'embeddings.npy'
                if emb_file.exists():
                    embeddings = np.load(emb_file)
                    for emb in embeddings:
                        face_db.append({
                            'id': member_id,
                            'name': name,
                            'feature': emb.reshape(1, -1).astype(np.float32)
                        })
                    print(f"  ✅ Loaded {len(embeddings)} embeddings for {name}")
                    continue
                    
            except Exception as e:
                print(f"Error loading embeddings for {person_dir}: {e}")
    
    if face_db:
        print(f"✅ Database loaded from embeddings: {len(face_db)} face embeddings")
        return
    
    # 3. Fallback: Load from augmented images (slower)
    for person_dir in TRAINING_DATA_PATH.glob('*'):
        if person_dir.is_dir():
            try:
                # Parse ID and Name
                parts = person_dir.name.split('_', 1)
                if len(parts) < 2: continue
                member_id = int(parts[0])
                name = parts[1].replace('_', ' ')
                
                # Load augmented images
                aug_path = person_dir / 'augmented'
                if aug_path.exists():
                    images = list(aug_path.glob('*.jpg'))
                    # Limit to 20 images per person to save RAM/Time if needed
                    # images = images[:20] 
                    
                    for img_file in images:
                        process_db_image(str(img_file), member_id, name)
            except Exception as e:
                print(f"Error loading {person_dir}: {e}")

    # 2. Load from face_database (Legacy)
    for img_file in FACE_DATABASE_PATH.glob('*.jpg'):
        try:
            filename = img_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                member_id = int(parts[0])
                name = parts[1]
                process_db_image(str(img_file), member_id, name)
        except Exception as e:
            print(f"Error loading legacy {img_file}: {e}")
            
    print(f"✅ Database loaded: {len(face_db)} face embeddings")

def process_db_image(img_path, member_id, name):
    """Process a single database image and add embedding to memory"""
    try:
        img = cv2.imread(img_path)
        if img is None: return

        # Detect face (assume cropped or simple detection)
        # For database images, we assume they are already good or cropped
        # But SFace needs aligned crop. 
        # SFace alignCrop needs detection result.
        
        # Simple detection for alignment
        height, width = img.shape[:2]
        # Create a dummy detection if it's already a crop (which augmented images are)
        # But SFace expects specific alignment.
        # Let's run detection on it to be safe.
        
        # Use YOLO to detect face in the training image
        if USE_YOLO:
            detection = yolo_detector.detect_largest_face(img, return_crop=False)
            if detection:
                x, y, w, h, conf = detection
                # SFace expects: face_box, face_landmarks (optional)
                # We can just pass the crop to feature() if aligned? 
                # No, feature() takes aligned image.
                # alignCrop() takes original image and box.
                
                # Construct face box for SFace: [x, y, w, h]
                face_box = np.array([x, y, w, h], dtype=np.int32)
                
                # Align and Crop
                aligned_face = recognizer.alignCrop(img, face_box)
                
                # Get Feature
                feature = recognizer.feature(aligned_face)
                
                face_db.append({
                    'id': member_id,
                    'name': name,
                    'feature': feature
                })
    except Exception as e:
        pass

def match_face(query_feature, query_image=None):
    """
    IMPROVED: Use L2 distance to MEAN embeddings only (not all 468 augmented ones).
    This is more accurate when people look similar.
    """
    if not face_db_means:
        return None, 0.0
    
    # Normalize query
    query_flat = query_feature.flatten()
    query_norm = query_flat / (np.linalg.norm(query_flat) + 1e-6)
    
    print(f"\n🔍 Matching against {len(face_db_means)} persons (mean embeddings)...")
    
    # === STEP 1: L2 distance to each person's MEAN embedding ===
    l2_distances = []
    cosine_sims = []
    
    for person_key, data in face_db_means.items():
        mean_emb = data['mean_embedding']
        mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        
        # L2 distance (lower = better)
        l2_dist = np.linalg.norm(query_flat - mean_emb)
        
        # Cosine similarity (higher = better)
        cos_sim = np.dot(query_norm, mean_norm)
        
        l2_distances.append({
            'id': data['id'],
            'name': data['name'],
            'person_key': person_key,
            'l2_dist': l2_dist,
            'cos_sim': cos_sim
        })
        cosine_sims.append((person_key, cos_sim))
    
    # Sort by L2 distance (ascending - lower is better)
    l2_distances.sort(key=lambda x: x['l2_dist'])
    
    print("  📊 L2 distances to mean embeddings:")
    for r in l2_distances:
        print(f"     {r['name']}: L2={r['l2_dist']:.3f}, cos={r['cos_sim']:.3f}")
    
    # === STEP 2: LBPH matching ===
    lbph_scores = {}
    if USE_LBPH and lbph_data is not None and query_image is not None:
        try:
            if len(query_image.shape) == 3:
                gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = query_image
            
            gray_resized = cv2.resize(gray, (128, 128))
            gray_eq = cv2.equalizeHist(gray_resized)
            
            # Create LBPH histogram for query
            temp_lbph = cv2.face.LBPHFaceRecognizer_create(
                radius=lbph_data['params']['radius'],
                neighbors=lbph_data['params']['neighbors'],
                grid_x=lbph_data['params']['grid_x'],
                grid_y=lbph_data['params']['grid_y']
            )
            temp_lbph.train([gray_eq], np.array([0]))
            query_hist = temp_lbph.getHistograms()[0].flatten().astype(np.float32)
            
            # Compare with each person using histogram correlation
            for person_name, person_data in lbph_data['persons'].items():
                mean_hist = person_data['mean_histogram'].astype(np.float32)
                corr = cv2.compareHist(query_hist, mean_hist, cv2.HISTCMP_CORREL)
                lbph_scores[person_name] = corr
            
            print("  🔷 LBPH correlations:")
            for pn, sc in sorted(lbph_scores.items(), key=lambda x: -x[1]):
                print(f"     {pn}: {sc:.3f}")
                
        except Exception as e:
            print(f"  ⚠️ LBPH error: {e}")
    
    # === STEP 3: Combined decision ===
    best = l2_distances[0]
    second = l2_distances[1] if len(l2_distances) > 1 else None
    
    # L2 thresholds (SFace typical range: 0.8-1.5 for same person)
    L2_THRESHOLD = 1.3  # Max L2 distance to accept as match
    L2_MARGIN = 0.08    # Required margin between best and second
    
    if best['l2_dist'] > L2_THRESHOLD:
        print(f"  ❌ L2 distance {best['l2_dist']:.3f} > threshold {L2_THRESHOLD}")
        return None, 0.0
    
    if best['cos_sim'] < 0.50:
        print(f"  ❌ Cosine similarity {best['cos_sim']:.3f} too low")
        return None, 0.0
    
    # Check margin
    if second:
        margin = second['l2_dist'] - best['l2_dist']
        print(f"  📏 L2 margin: {margin:.3f} ({best['name']} vs {second['name']})")
        
        if margin < L2_MARGIN:
            print(f"  ⚠️ Margin too small, using LBPH tiebreaker...")
            
            # Use LBPH to break the tie
            best_lbph = lbph_scores.get(best['person_key'], 0)
            second_lbph = lbph_scores.get(second['person_key'], 0)
            
            if second_lbph > best_lbph + 0.05:
                print(f"  🔄 LBPH prefers {second['name']} ({second_lbph:.3f} vs {best_lbph:.3f})")
                best = second
            else:
                print(f"  ✓ LBPH confirms {best['name']} ({best_lbph:.3f} vs {second_lbph:.3f})")
    
    # Convert L2 to confidence score (lower L2 = higher confidence)
    confidence = max(0, 1.0 - best['l2_dist'] / 2.0)
    
    print(f"  ✅ Match: {best['name']} (L2={best['l2_dist']:.3f}, conf={confidence:.3f})")
    return {'id': best['id'], 'name': best['name']}, confidence
    
    # Thresholds
    MATCH_THRESHOLD = 0.25
    
    if best['score'] < MATCH_THRESHOLD:
        print(f"  ❌ Score {best['score']:.3f} below threshold {MATCH_THRESHOLD}")
        return None, best['score']
    
    if best['best'] < 0.30:
        print(f"  ❌ Best similarity {best['best']:.3f} too low - unknown person")
        return None, best['score']
    
    # Margin check with LBPH as PRIMARY tiebreaker
    if len(results) > 1:
        second = results[1]
        margin = best['score'] - second['score']
        print(f"  📏 Margin: {margin:.3f} ({best['name']} vs {second['name']})")
        
        if margin < 0.03:
            print(f"  ⚠️ Close match! Using LBPH as tiebreaker...")
            # LBPH is most reliable for texture discrimination
            # Only switch if LBPH strongly favors the second candidate
            if second['lbph_score'] > best['lbph_score'] + 0.1:
                print(f"  🔄 LBPH tiebreaker: switching to {second['name']}")
                best = second
            # Don't use vote tiebreaker - it's less reliable than LBPH
            # elif second['votes'] > best['votes'] + 3:
    print(f"  ✅ Match: {best['name']} (L2={best['l2_dist']:.3f}, conf={confidence:.3f})")
    return {'id': best['id'], 'name': best['name']}, confidence

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'backend': 'opencv-sface'})

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[-1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 1. Detect Face (YOLO)
        face_box = None
        face_coords = None
        face_crop = None  # Original YOLO crop for LBPH
        
        if USE_YOLO:
            detection = yolo_detector.detect_largest_face(img_cv, return_crop=False)
            if detection:
                x, y, w, h, conf = detection
                face_box = np.array([x, y, w, h], dtype=np.int32)
                face_coords = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'confidence': float(conf)}
                # Extract face crop from original image for LBPH
                face_crop = img_cv[int(y):int(y+h), int(x):int(x+w)].copy()
        
        if face_box is None:
            return jsonify({'success': True, 'match': False, 'error': 'No face detected'})
            
        # 2. Recognize Face (SFace)
        aligned_face = recognizer.alignCrop(img_cv, face_box)
        query_feature = recognizer.feature(aligned_face)
        
        # Pass ORIGINAL face crop for LBPH (not the SFace aligned face)
        match, score = match_face(query_feature, face_crop)
        
        if match:
            return jsonify({
                'success': True,
                'match': True,
                'member_id': match['id'],
                'name': match['name'],
                'confidence': float(score),
                'distance': 1.0 - float(score), # Fake distance for compatibility
                'face_coords': face_coords
            })
        else:
            return jsonify({
                'success': True,
                'match': False,
                'confidence': float(score),
                'face_coords': face_coords
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/register', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        base64_image = data.get('image')
        angle = data.get('angle', 'center')
        
        if not all([member_id, name, base64_image]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
            
        # Create directory structure: training_data/{id}_{name}/raw
        safe_name = name.replace(' ', '_')
        person_dir = TRAINING_DATA_PATH / f"{member_id}_{safe_name}"
        raw_dir = person_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Decode and save image
        if 'base64,' in base64_image:
            base64_image = base64_image.split('base64,')[1]
            
        img_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(img_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        filename = f"{angle}.jpg"
        save_path = raw_dir / filename
        img.save(save_path, 'JPEG', quality=95)
        
        print(f"✅ Saved registration image: {save_path}")
        return jsonify({'success': True, 'path': str(save_path)})
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        augmentations = data.get('augmentations', 10)
        
        if not member_id or not name:
            # If no ID provided, just reload DB (legacy behavior)
            load_database()
            return jsonify({'success': True, 'message': 'Database reloaded'})

        # Run augmentation script
        print(f"🔄 Starting training for {name} (ID: {member_id})...")
        cmd = f'python train_person.py --id {member_id} --name "{name}" -n {augmentations}'
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            return jsonify({'success': False, 'error': 'Training script failed'}), 500
        
        # Reload database to include new person
        load_database()
        
        # Count images
        person_dir = TRAINING_DATA_PATH / f"{member_id}_{name.replace(' ', '_')}"
        aug_dir = person_dir / 'augmented'
        total_images = len(list(aug_dir.glob('*.jpg'))) if aug_dir.exists() else 0
        
        return jsonify({
            'success': True, 
            'message': 'Training complete',
            'total_images': total_images
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        
        if not member_id or not name:
            return jsonify({'success': False, 'error': 'Missing member_id or name'}), 400
            
        # Construct folder path
        safe_name = name.replace(' ', '_')
        person_dir = TRAINING_DATA_PATH / f"{member_id}_{safe_name}"
        
        if person_dir.exists() and person_dir.is_dir():
            import shutil
            shutil.rmtree(person_dir)
            print(f"✅ Deleted training data for {name} (ID: {member_id})")
            
            # Reload database to remove embeddings from memory
            load_database()
            return jsonify({'success': True, 'message': 'Training data deleted'})
        else:
            return jsonify({'success': False, 'error': 'Folder not found'}), 404
            
    except Exception as e:
        print(f"❌ Deletion error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Load DB on startup
    load_database()
    # Use PORT from environment variable for cloud hosting
    port = int(os.environ.get('PORT', 5001))
    print(f"🚀 Face Recognition API (YOLO + SFace) running on port {port}")
    app.run(host='0.0.0.0', port=port)
