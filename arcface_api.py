"""
ArcFace Face Recognition API using InsightFace
Uses buffalo_l model which has better accuracy than SFace
"""
import os
import cv2
import numpy as np
import base64
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
import time

print("🚀 Initializing ArcFace Face Recognition API...")

# Initialize InsightFace with buffalo_l model
print("Loading ArcFace model (buffalo_l)...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ ArcFace model loaded!")

# Flask app
app = Flask(__name__)
CORS(app)

# Paths
TRAINING_DATA_PATH = Path('training_data')
RECOGNITION_DB_PATH = Path('recognition_database_arcface.pkl')
LBPH_DB_PATH = Path('lbph_database.pkl')

# In-memory database
face_db_means = {}  # person_key -> {'id': int, 'name': str, 'mean_embedding': np.array}

# Load LBPH for texture tiebreaker (optional - can be slow)
USE_LBPH = False  # Disabled for now - ArcFace is accurate enough
lbph_data = None
# Uncomment to enable LBPH:
# if LBPH_DB_PATH.exists():
#     try:
#         print("🔄 Loading LBPH database...")
#         with open(LBPH_DB_PATH, 'rb') as f:
#             lbph_data = pickle.load(f)
#         USE_LBPH = True
#         print(f"✅ LBPH loaded: {len(lbph_data['persons'])} persons")
#     except Exception as e:
#         print(f"⚠️ LBPH load failed: {e}")


def load_database():
    """Load ArcFace embeddings database"""
    global face_db_means
    face_db_means = {}
    
    print("🔄 Loading ArcFace database...")
    
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
                
                face_db_means[person_key] = {
                    'id': member_id,
                    'name': name,
                    'mean_embedding': person_data['mean_embedding'],
                    'embeddings': person_data.get('embeddings', [])
                }
            
            print(f"✅ Loaded ArcFace database: {len(face_db_means)} persons")
            return
        except Exception as e:
            print(f"⚠️ Failed to load ArcFace DB: {e}")
    
    print("⚠️ No ArcFace database found. Run build_arcface_db.py first!")


def match_face(query_embedding, query_image=None):
    """
    Match face using ArcFace embeddings with L2 distance.
    Uses LBPH as tiebreaker when needed.
    """
    if not face_db_means:
        return None, 0.0
    
    query_flat = query_embedding.flatten()
    query_norm = query_flat / (np.linalg.norm(query_flat) + 1e-6)
    
    print(f"\n🔍 ArcFace matching against {len(face_db_means)} persons...")
    
    # Calculate distances to each person
    distances = []
    for person_key, data in face_db_means.items():
        mean_emb = data['mean_embedding'].flatten()
        mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
        
        # Cosine similarity (higher = better)
        cos_sim = np.dot(query_norm, mean_norm)
        
        # L2 distance (lower = better)
        l2_dist = np.linalg.norm(query_flat - mean_emb)
        
        distances.append({
            'id': data['id'],
            'name': data['name'],
            'person_key': person_key,
            'cos_sim': cos_sim,
            'l2_dist': l2_dist
        })
    
    # Sort by cosine similarity (descending)
    distances.sort(key=lambda x: x['cos_sim'], reverse=True)
    
    print("  📊 ArcFace similarities:")
    for r in distances:
        print(f"     {r['name']}: cos={r['cos_sim']:.4f}, L2={r['l2_dist']:.3f}")
    
    # LBPH tiebreaker
    lbph_scores = {}
    if USE_LBPH and lbph_data is not None and query_image is not None:
        try:
            if len(query_image.shape) == 3:
                gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = query_image
            
            gray_resized = cv2.resize(gray, (128, 128))
            gray_eq = cv2.equalizeHist(gray_resized)
            
            temp_lbph = cv2.face.LBPHFaceRecognizer_create(
                radius=lbph_data['params']['radius'],
                neighbors=lbph_data['params']['neighbors'],
                grid_x=lbph_data['params']['grid_x'],
                grid_y=lbph_data['params']['grid_y']
            )
            temp_lbph.train([gray_eq], np.array([0]))
            query_hist = temp_lbph.getHistograms()[0].flatten().astype(np.float32)
            
            for person_name, person_data in lbph_data['persons'].items():
                mean_hist = person_data['mean_histogram'].astype(np.float32)
                corr = cv2.compareHist(query_hist, mean_hist, cv2.HISTCMP_CORREL)
                lbph_scores[person_name] = corr
            
            print("  🔷 LBPH correlations:")
            for pn, sc in sorted(lbph_scores.items(), key=lambda x: -x[1]):
                print(f"     {pn}: {sc:.3f}")
                
        except Exception as e:
            print(f"  ⚠️ LBPH error: {e}")
    
    best = distances[0]
    second = distances[1] if len(distances) > 1 else None
    
    # Thresholds for ArcFace (cosine similarity)
    MATCH_THRESHOLD = 0.55  # Minimum similarity to be a match (correct matches: 0.586-0.692)
    MARGIN_THRESHOLD = 0.05  # Minimum margin between best and second
    
    if best['cos_sim'] < MATCH_THRESHOLD:
        print(f"  ❌ Best similarity {best['cos_sim']:.3f} < threshold {MATCH_THRESHOLD}")
        return None, 0.0
    
    # Check margin
    if second:
        margin = best['cos_sim'] - second['cos_sim']
        print(f"  📏 Margin: {margin:.4f} ({best['name']} vs {second['name']})")
        
        if margin < MARGIN_THRESHOLD:
            print(f"  ⚠️ Close match, using LBPH tiebreaker...")
            
            best_lbph = lbph_scores.get(best['person_key'], 0)
            second_lbph = lbph_scores.get(second['person_key'], 0)
            
            if second_lbph > best_lbph + 0.05:
                print(f"  🔄 LBPH prefers {second['name']} ({second_lbph:.3f} vs {best_lbph:.3f})")
                best = second
            else:
                print(f"  ✓ LBPH confirms {best['name']}")
    
    confidence = best['cos_sim']
    print(f"  ✅ Match: {best['name']} (similarity={confidence:.4f})")
    
    return {'id': best['id'], 'name': best['name']}, confidence


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'backend': 'arcface-buffalo_l'})


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
        
        # Detect and analyze face with InsightFace
        faces = face_app.get(img_cv)
        
        if not faces:
            return jsonify({'success': True, 'match': False, 'error': 'No face detected'})
        
        # Get largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        # Get face box for LBPH
        bbox = face.bbox.astype(int)
        x, y, x2, y2 = bbox
        face_crop = img_cv[y:y2, x:x2].copy()
        
        face_coords = {
            'x': int(x),
            'y': int(y),
            'w': int(x2 - x),
            'h': int(y2 - y),
            'confidence': float(face.det_score)
        }
        
        # Get ArcFace embedding
        query_embedding = face.embedding
        
        # Match
        match, score = match_face(query_embedding, face_crop)
        
        if match:
            return jsonify({
                'success': True,
                'match': True,
                'member_id': match['id'],
                'name': match['name'],
                'confidence': float(score),
                'distance': 1.0 - float(score),
                'face_coords': face_coords
            })
        else:
            return jsonify({
                'success': True,
                'match': False,
                'confidence': float(score) if score else 0.0,
                'face_coords': face_coords
            })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def register_face():
    """Register a new face.

    Expects JSON body with keys: `member_id` (int), `name` (str), `image` (base64 data URL or raw base64).
    This will:
    - create training_data/<id>_<name>/raw and /cropped folders
    - save the original image and a cropped face image (if detected)
    - compute ArcFace embedding and append/update the ArcFace DB
    - update in-memory `face_db_means`
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body provided'}), 400

        member_id = data.get('member_id')
        name = data.get('name')
        image_b64 = data.get('image')

        if member_id is None or name is None or image_b64 is None:
            return jsonify({'success': False, 'error': 'Required fields: member_id, name, image'}), 400

        # Normalize name and paths
        safe_name = str(name).strip().replace(' ', '_')
        person_dir = TRAINING_DATA_PATH / f"{int(member_id)}_{safe_name}"
        raw_dir = person_dir / 'raw'
        cropped_dir = person_dir / 'cropped'
        raw_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        # Decode image (allow data URLs)
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[-1]
        img_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Save original image
        ts = int(time.time())
        raw_path = raw_dir / f"reg_{ts}.jpg"
        cv2.imwrite(str(raw_path), img_cv)

        # Detect face and save cropped face if possible
        faces = face_app.get(img_cv)
        saved_crop = None
        embedding = None
        if faces:
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            x = max(0, x); y = max(0, y); x2 = max(x2, x+1); y2 = max(y2, y+1)
            face_crop = img_cv[y:y2, x:x2].copy()
            if face_crop.size != 0:
                crop_path = cropped_dir / f"crop_{ts}.jpg"
                cv2.imwrite(str(crop_path), face_crop)
                saved_crop = str(crop_path)
                # Get embedding
                embedding = face.embedding

        # If no face detected, return error
        if embedding is None:
            return jsonify({'success': False, 'error': 'No face detected in provided image'}), 400

        # Load or create DB
        if RECOGNITION_DB_PATH.exists():
            with open(RECOGNITION_DB_PATH, 'rb') as f:
                db = pickle.load(f)
        else:
            db = {'persons': {}, 'model': 'arcface-buffalo_l'}

        person_key = f"{int(member_id)}_{safe_name}"
        if person_key in db['persons']:
            prev = db['persons'][person_key]['embeddings']
            try:
                prev_arr = np.array(prev)
                new_arr = np.vstack([prev_arr, embedding])
            except Exception:
                # Fallback if previous is a list
                new_arr = np.vstack([np.array(prev), embedding])
        else:
            new_arr = np.array([embedding])

        mean_emb = np.mean(new_arr, axis=0)
        db['persons'][person_key] = {
            'embeddings': new_arr,
            'mean_embedding': mean_emb
        }

        # Save DB
        with open(RECOGNITION_DB_PATH, 'wb') as f:
            pickle.dump(db, f)

        # Update in-memory index
        face_db_means[person_key] = {
            'id': int(member_id),
            'name': name.strip(),
            'mean_embedding': mean_emb,
            'embeddings': new_arr
        }

        return jsonify({
            'success': True,
            'message': 'Member registered',
            'member_key': person_key,
            'raw_path': str(raw_path),
            'crop_path': saved_crop,
            'embeddings_count': int(new_arr.shape[0])
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# Load database at startup
load_database()


if __name__ == '__main__':
    print("🚀 ArcFace Face Recognition API running on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
