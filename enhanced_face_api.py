"""
Enhanced Face Recognition API with Advanced Features
=====================================================
Combines YOLO face detection with:
- SFace/ArcFace deep embeddings (512D)
- Geometric facial measurements (eye distance, jaw shape, etc.)
- Facial landmark analysis (68+ points)
- Texture pattern analysis (LBP)

This provides robust face recognition using multiple feature types.
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
from PIL import Image
import time

# Import YOLO detector
try:
    from face_detector_yolo import YOLOFaceDetector
    USE_YOLO = True
    print("✅ YOLOv11 detector loaded")
except Exception as e:
    USE_YOLO = False
    print(f"⚠️ YOLOv11 not available: {e}")

# Import Advanced Features
try:
    from advanced_face_features import AdvancedFaceFeatures, HybridFaceRecognizer
    USE_ADVANCED = True
    print("✅ Advanced face features module loaded")
except Exception as e:
    USE_ADVANCED = False
    print(f"⚠️ Advanced features not available: {e}")

app = Flask(__name__)
CORS(app)

# Paths
TRAINING_DATA_PATH = Path('training_data')
TRAINING_DATA_PATH.mkdir(exist_ok=True)
FACE_DATABASE_PATH = Path('face_database')
FACE_DATABASE_PATH.mkdir(exist_ok=True)
MODEL_PATH = "face_recognition_sface_2021dec.onnx"

# Initialize Face Recognizer (SFace - ArcFace style)
recognizer = cv2.FaceRecognizerSF.create(
    model=MODEL_PATH,
    config="",
    backend_id=0,
    target_id=0
)
print("✅ SFace Recognizer initialized (ArcFace-style embeddings)")

# Initialize YOLO Detector
if USE_YOLO:
    yolo_detector = YOLOFaceDetector(confidence_threshold=0.3)
    print("✅ YOLOv11 face detector initialized")

# Initialize Advanced Feature Extractor
if USE_ADVANCED:
    try:
        advanced_features = AdvancedFaceFeatures()
        hybrid_recognizer = HybridFaceRecognizer(MODEL_PATH)
        print("✅ Advanced geometric features enabled")
    except Exception as e:
        USE_ADVANCED = False
        print(f"⚠️ Advanced features initialization failed: {e}")

# In-memory database for fast matching
face_db = []

# Configuration
CONFIG = {
    'use_geometric_features': USE_ADVANCED,
    'deep_weight': 0.7,  # Weight for deep embeddings
    'geometric_weight': 0.3,  # Weight for geometric features
    'cosine_threshold': 0.35,  # Minimum cosine similarity for match
    'combined_threshold': 0.40,  # Threshold when using combined features
}


def load_database():
    """Load all faces from training_data into memory with advanced features."""
    global face_db
    face_db = []
    print("🔄 Loading face database with advanced features...")
    
    # 1. Load from training_data (New structure)
    for person_dir in TRAINING_DATA_PATH.glob('*'):
        if person_dir.is_dir():
            try:
                # Parse ID and Name
                parts = person_dir.name.split('_', 1)
                if len(parts) < 2:
                    continue
                member_id = int(parts[0])
                name = parts[1].replace('_', ' ')
                
                # Load augmented images
                aug_path = person_dir / 'augmented'
                if aug_path.exists():
                    images = list(aug_path.glob('*.jpg'))
                    
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
    
    # Print feature breakdown
    with_geometric = sum(1 for e in face_db if e.get('geometric_vector') is not None)
    print(f"   - Deep embeddings: {len(face_db)}")
    print(f"   - With geometric features: {with_geometric}")


def process_db_image(img_path, member_id, name):
    """Process a single database image and add embeddings to memory."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return

        # Detect face with YOLO
        if USE_YOLO:
            detection = yolo_detector.detect_largest_face(img, return_crop=False)
            if not detection:
                return
                
            x, y, w, h, conf = detection
            face_box = np.array([x, y, w, h], dtype=np.int32)
            
            # Get deep embedding (SFace/ArcFace style)
            aligned_face = recognizer.alignCrop(img, face_box)
            deep_feature = recognizer.feature(aligned_face)
            
            entry = {
                'id': member_id,
                'name': name,
                'feature': deep_feature,
                'geometric_features': None,
                'geometric_vector': None,
            }
            
            # Get geometric features if available
            if USE_ADVANCED:
                try:
                    features_dict, feature_vector = advanced_features.extract_all_features(img, (x, y, w, h))
                    entry['geometric_features'] = features_dict
                    entry['geometric_vector'] = feature_vector
                except Exception as e:
                    pass  # Geometric features are optional
            
            face_db.append(entry)
            
    except Exception as e:
        pass


def match_face(query_deep_feature, query_geometric=None):
    """
    Find best match for a query face using hybrid matching.
    
    Combines:
    - Deep embedding cosine similarity (primary)
    - Geometric feature similarity (secondary)
    """
    best_score = 0.0
    best_match = None
    match_details = {}
    
    for entry in face_db:
        # Deep embedding score (SFace cosine similarity)
        deep_score = recognizer.match(
            query_deep_feature, 
            entry['feature'], 
            cv2.FaceRecognizerSF_FR_COSINE
        )
        
        # Geometric feature score (if available)
        geometric_score = 0.0
        if CONFIG['use_geometric_features'] and query_geometric is not None and entry['geometric_vector'] is not None:
            try:
                geometric_score = advanced_features.compare_features(
                    query_geometric, 
                    entry['geometric_vector']
                )
            except:
                pass
        
        # Combined score
        if geometric_score > 0:
            combined_score = (
                CONFIG['deep_weight'] * deep_score + 
                CONFIG['geometric_weight'] * geometric_score
            )
        else:
            combined_score = deep_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = entry
            match_details = {
                'deep_score': float(deep_score),
                'geometric_score': float(geometric_score),
                'combined_score': float(combined_score),
            }
    
    # Dynamic threshold based on feature availability
    threshold = CONFIG['combined_threshold'] if query_geometric is not None else CONFIG['cosine_threshold']
    
    if best_score > threshold:
        return best_match, best_score, match_details
        
    return None, best_score, match_details


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'backend': 'yolo-sface-advanced',
        'features': {
            'yolo_detection': USE_YOLO,
            'deep_embeddings': True,
            'geometric_features': USE_ADVANCED,
        },
        'database_size': len(face_db),
    })


@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Verify a face against the database.
    
    Uses hybrid matching:
    1. YOLO for fast face detection
    2. SFace for deep embeddings (ArcFace-style)
    3. Geometric features for additional accuracy
    """
    try:
        start_time = time.time()
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[-1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 1. Detect Face (YOLO - fast)
        face_box = None
        face_coords = None
        
        if USE_YOLO:
            detection = yolo_detector.detect_largest_face(img_cv, return_crop=False)
            if detection:
                x, y, w, h, conf = detection
                face_box = np.array([x, y, w, h], dtype=np.int32)
                face_coords = {
                    'x': int(x), 'y': int(y), 
                    'w': int(w), 'h': int(h), 
                    'confidence': float(conf)
                }
        
        if face_box is None:
            return jsonify({
                'success': True, 
                'match': False, 
                'error': 'No face detected'
            })
            
        # 2. Extract Deep Embedding (SFace/ArcFace)
        aligned_face = recognizer.alignCrop(img_cv, face_box)
        query_deep_feature = recognizer.feature(aligned_face)
        
        # 3. Extract Geometric Features (optional but improves accuracy)
        query_geometric = None
        geometric_details = None
        
        if USE_ADVANCED:
            try:
                x, y, w, h = face_box
                features_dict, feature_vector = advanced_features.extract_all_features(
                    img_cv, (x, y, w, h)
                )
                query_geometric = feature_vector
                geometric_details = features_dict
            except Exception as e:
                print(f"Geometric extraction failed: {e}")
        
        # 4. Match against database
        match, score, match_details = match_face(query_deep_feature, query_geometric)
        
        processing_time = time.time() - start_time
        
        if match:
            # Build detailed response
            response = {
                'success': True,
                'match': True,
                'member_id': match['id'],
                'name': match['name'],
                'confidence': float(score),
                'distance': 1.0 - float(score),
                'face_coords': face_coords,
                'processing_time_ms': int(processing_time * 1000),
                'match_details': {
                    'deep_embedding_score': match_details.get('deep_score', 0),
                    'geometric_score': match_details.get('geometric_score', 0),
                    'combined_score': match_details.get('combined_score', 0),
                    'method': 'hybrid' if match_details.get('geometric_score', 0) > 0 else 'deep_only'
                }
            }
            
            # Add geometric analysis if available
            if geometric_details:
                response['face_analysis'] = {
                    'eye_distance_ratio': geometric_details.get('left_eye_width_ratio', 0),
                    'face_aspect_ratio': geometric_details.get('face_aspect_ratio', 0),
                    'jaw_angle': geometric_details.get('chin_angle', 0),
                    'nose_ratio': geometric_details.get('nose_bridge_length_ratio', 0),
                    'symmetry_score': geometric_details.get('horizontal_symmetry', 0),
                }
            
            return jsonify(response)
        else:
            return jsonify({
                'success': True,
                'match': False,
                'confidence': float(score),
                'face_coords': face_coords,
                'processing_time_ms': int(processing_time * 1000),
                'match_details': match_details
            })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_face():
    """
    Analyze a face and return detailed geometric features.
    
    Features returned:
    - Eye measurements (distance, aspect ratio, symmetry)
    - Nose structure (bridge length, width)
    - Jaw shape (width, angles)
    - Cheekbone analysis
    - Facial proportions
    - Symmetry scores
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        # Decode image
        img_data = base64.b64decode(data['image'].split(',')[-1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Detect face
        if USE_YOLO:
            detection = yolo_detector.detect_largest_face(img_cv, return_crop=False)
            if not detection:
                return jsonify({'success': False, 'error': 'No face detected'})
                
            x, y, w, h, conf = detection
            face_box = (x, y, w, h)
        else:
            face_box = None
        
        # Extract advanced features
        if not USE_ADVANCED:
            return jsonify({
                'success': False, 
                'error': 'Advanced features not available. Install dlib.'
            })
        
        features_dict, feature_vector = advanced_features.extract_all_features(img_cv, face_box)
        
        if features_dict is None:
            return jsonify({'success': False, 'error': 'Could not extract features'})
        
        # Organize features by category
        analysis = {
            'success': True,
            'face_detected': True,
            'face_coords': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            
            'eye_features': {
                'inter_eye_distance': features_dict.get('inter_eye_distance', 0),
                'left_eye_width_ratio': features_dict.get('left_eye_width_ratio', 0),
                'right_eye_width_ratio': features_dict.get('right_eye_width_ratio', 0),
                'left_eye_aspect_ratio': features_dict.get('left_eye_aspect_ratio', 0),
                'right_eye_aspect_ratio': features_dict.get('right_eye_aspect_ratio', 0),
                'eye_symmetry': features_dict.get('eye_size_symmetry', 0),
                'eye_tilt_angle': features_dict.get('eye_tilt_angle', 0),
            },
            
            'nose_features': {
                'bridge_length_ratio': features_dict.get('nose_bridge_length_ratio', 0),
                'width_ratio': features_dict.get('nose_width_ratio', 0),
                'base_width_ratio': features_dict.get('nose_base_width_ratio', 0),
                'symmetry': features_dict.get('nose_symmetry', 0),
            },
            
            'jaw_features': {
                'jaw_width_ratio': features_dict.get('jaw_width_ratio', 0),
                'face_height_ratio': features_dict.get('face_height_ratio', 0),
                'face_aspect_ratio': features_dict.get('face_aspect_ratio', 0),
                'chin_angle': features_dict.get('chin_angle', 0),
                'left_jaw_angle': features_dict.get('left_jaw_angle', 0),
                'right_jaw_angle': features_dict.get('right_jaw_angle', 0),
                'symmetry': features_dict.get('jaw_symmetry', 0),
            },
            
            'cheekbone_features': {
                'width_ratio': features_dict.get('cheekbone_width_ratio', 0),
                'left_angle': features_dict.get('left_cheekbone_angle', 0),
                'right_angle': features_dict.get('right_cheekbone_angle', 0),
                'symmetry': features_dict.get('cheekbone_symmetry', 0),
            },
            
            'mouth_features': {
                'width_ratio': features_dict.get('mouth_width_ratio', 0),
                'upper_lip_ratio': features_dict.get('upper_lip_ratio', 0),
                'lower_lip_ratio': features_dict.get('lower_lip_ratio', 0),
                'lip_ratio': features_dict.get('lip_ratio', 0),
                'mouth_nose_distance': features_dict.get('mouth_nose_distance_ratio', 0),
            },
            
            'proportions': {
                'upper_third_deviation': features_dict.get('upper_third_deviation', 0),
                'middle_third_deviation': features_dict.get('middle_third_deviation', 0),
                'lower_third_deviation': features_dict.get('lower_third_deviation', 0),
                'horizontal_symmetry': features_dict.get('horizontal_symmetry', 0),
            },
            
            'feature_vector_size': len(feature_vector) if feature_vector is not None else 0,
        }
        
        return jsonify(analysis)

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def register_face():
    """Register a new face image for training."""
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        base64_image = data.get('image')
        angle = data.get('angle', 'center')
        
        if not all([member_id, name, base64_image]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
            
        # Create directory structure
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
    """Train/augment images for a person and reload database."""
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        augmentations = data.get('augmentations', 10)
        
        if not member_id or not name:
            load_database()
            return jsonify({'success': True, 'message': 'Database reloaded'})

        print(f"🔄 Starting training for {name} (ID: {member_id})...")
        cmd = f'python train_person.py --id {member_id} --name "{name}" -n {augmentations}'
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            return jsonify({'success': False, 'error': 'Training script failed'}), 500
        
        load_database()
        
        person_dir = TRAINING_DATA_PATH / f"{member_id}_{name.replace(' ', '_')}"
        aug_dir = person_dir / 'augmented'
        total_images = len(list(aug_dir.glob('*.jpg'))) if aug_dir.exists() else 0
        
        return jsonify({
            'success': True, 
            'message': 'Training complete with advanced features',
            'total_images': total_images,
            'features_enabled': {
                'deep_embeddings': True,
                'geometric_features': USE_ADVANCED
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete_person', methods=['POST'])
def delete_person():
    """Delete training data for a person."""
    try:
        data = request.get_json()
        member_id = data.get('member_id')
        name = data.get('name')
        
        if not member_id or not name:
            return jsonify({'success': False, 'error': 'Missing member_id or name'}), 400
            
        safe_name = name.replace(' ', '_')
        person_dir = TRAINING_DATA_PATH / f"{member_id}_{safe_name}"
        
        if person_dir.exists() and person_dir.is_dir():
            import shutil
            shutil.rmtree(person_dir)
            print(f"✅ Deleted training data for {name} (ID: {member_id})")
            load_database()
            return jsonify({'success': True, 'message': 'Training data deleted'})
        else:
            return jsonify({'success': False, 'error': 'Folder not found'}), 404
            
    except Exception as e:
        print(f"❌ Deletion error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or update recognition configuration."""
    if request.method == 'GET':
        return jsonify(CONFIG)
    
    try:
        data = request.get_json()
        
        if 'deep_weight' in data:
            CONFIG['deep_weight'] = float(data['deep_weight'])
            CONFIG['geometric_weight'] = 1.0 - CONFIG['deep_weight']
            
        if 'cosine_threshold' in data:
            CONFIG['cosine_threshold'] = float(data['cosine_threshold'])
            
        if 'combined_threshold' in data:
            CONFIG['combined_threshold'] = float(data['combined_threshold'])
            
        if 'use_geometric_features' in data:
            CONFIG['use_geometric_features'] = bool(data['use_geometric_features']) and USE_ADVANCED
            
        return jsonify({'success': True, 'config': CONFIG})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Enhanced Face Recognition API")
    print("="*60)
    print("\nFeatures:")
    print("  ✅ YOLO v8/v11 Face Detection")
    print("  ✅ SFace Deep Embeddings (ArcFace-style)")
    if USE_ADVANCED:
        print("  ✅ Geometric Feature Analysis:")
        print("     - Eye distance & aspect ratios")
        print("     - Jaw shape & angles")
        print("     - Cheekbone measurements")
        print("     - Nose structure")
        print("     - Facial proportions")
        print("     - Symmetry scores")
    else:
        print("  ⚠️ Geometric features disabled (install dlib)")
    print("="*60 + "\n")
    
    load_database()
    
    print(f"\n🌐 API running on http://0.0.0.0:5001")
    print("   Endpoints:")
    print("   - GET  /health  - Check status")
    print("   - POST /verify  - Verify a face")
    print("   - POST /analyze - Get detailed face analysis")
    print("   - POST /register - Register new face")
    print("   - POST /train   - Train/augment faces")
    print("   - GET/POST /config - View/update settings")
    
    app.run(host='0.0.0.0', port=5001)
