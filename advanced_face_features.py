"""
Advanced Face Feature Extraction Module (OpenCV Only)
======================================================
Extracts geometric facial features using OpenCV's built-in tools.
No external dependencies like dlib or mediapipe required.

Features extracted:
- Deep embeddings (512D vector from SFace)
- Eye distance and aspect ratio
- Jaw shape and width approximation
- Nose structure estimation
- Facial proportions
- Symmetry scores
- Texture patterns (LBP)

Works with any face already detected by YOLO.
"""

import cv2
import numpy as np
import math
from pathlib import Path


class AdvancedFaceFeatures:
    """
    Extract advanced geometric and texture features from faces.
    Uses OpenCV's built-in tools (no dlib/mediapipe required).
    """
    
    def __init__(self):
        """Initialize feature extractors."""
        # OpenCV's face detector for alignment
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Try to load LBFModel for facial landmarks (if available)
        lbf_model = Path("lbfmodel.yaml")
        if lbf_model.exists():
            self.facemark = cv2.face.createFacemarkLBF()
            self.facemark.loadModel(str(lbf_model))
            self.USE_FACEMARK = True
            print("✅ LBF Facemark model loaded for 68 landmarks")
        else:
            self.USE_FACEMARK = False
            print("⚠️ LBF model not found, using eye-based estimation")
        
        print("✅ OpenCV-based feature extractor initialized")
    
    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
    
    def angle_between_points(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def detect_eyes(self, face_roi, face_box):
        """Detect eyes in face ROI and return their centers."""
        x, y, w, h = face_box
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(w*0.1), int(h*0.05)),
            maxSize=(int(w*0.4), int(h*0.3))
        )
        
        if len(eyes) < 2:
            # Estimate eye positions based on face proportions
            left_eye = (int(w * 0.3), int(h * 0.35))
            right_eye = (int(w * 0.7), int(h * 0.35))
            return left_eye, right_eye, False
        
        # Sort eyes by x coordinate (left to right)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Get centers of first two eyes
        left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
        right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
        
        return left_eye, right_eye, True
    
    def estimate_landmarks(self, face_roi, face_box):
        """
        Estimate key facial landmarks using eye detection and face proportions.
        Returns a dict of landmark positions relative to face ROI.
        """
        x, y, w, h = face_box
        
        # Detect eyes
        left_eye, right_eye, eyes_detected = self.detect_eyes(face_roi, face_box)
        
        # Calculate inter-eye distance
        inter_eye_dist = self.euclidean_distance(left_eye, right_eye)
        
        # Estimate other landmarks based on facial proportions
        # These are approximate positions based on average face geometry
        landmarks = {
            # Eyes
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_eye_outer': (int(left_eye[0] - w*0.08), left_eye[1]),
            'left_eye_inner': (int(left_eye[0] + w*0.08), left_eye[1]),
            'right_eye_outer': (int(right_eye[0] + w*0.08), right_eye[1]),
            'right_eye_inner': (int(right_eye[0] - w*0.08), right_eye[1]),
            
            # Nose (estimated from eye positions)
            'nose_bridge': ((left_eye[0] + right_eye[0])//2, int(h * 0.45)),
            'nose_tip': ((left_eye[0] + right_eye[0])//2, int(h * 0.60)),
            'left_nostril': (int(w * 0.38), int(h * 0.65)),
            'right_nostril': (int(w * 0.62), int(h * 0.65)),
            
            # Mouth
            'mouth_left': (int(w * 0.32), int(h * 0.78)),
            'mouth_right': (int(w * 0.68), int(h * 0.78)),
            'mouth_top': (int(w * 0.5), int(h * 0.73)),
            'mouth_bottom': (int(w * 0.5), int(h * 0.82)),
            
            # Jaw
            'chin': (int(w * 0.5), int(h * 0.95)),
            'left_jaw': (int(w * 0.1), int(h * 0.65)),
            'right_jaw': (int(w * 0.9), int(h * 0.65)),
            'left_cheek': (int(w * 0.15), int(h * 0.45)),
            'right_cheek': (int(w * 0.85), int(h * 0.45)),
            
            # Forehead
            'forehead_center': ((left_eye[0] + right_eye[0])//2, int(h * 0.15)),
        }
        
        return landmarks, inter_eye_dist, eyes_detected
    
    def extract_eye_features(self, landmarks, inter_eye_dist):
        """Extract eye-related geometric features."""
        features = {}
        
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        
        # Eye width approximations
        left_width = self.euclidean_distance(landmarks['left_eye_outer'], landmarks['left_eye_inner'])
        right_width = self.euclidean_distance(landmarks['right_eye_inner'], landmarks['right_eye_outer'])
        
        # Normalized by inter-eye distance
        features['inter_eye_distance'] = 1.0  # Reference
        features['left_eye_width_ratio'] = left_width / (inter_eye_dist + 1e-6)
        features['right_eye_width_ratio'] = right_width / (inter_eye_dist + 1e-6)
        
        # Eye symmetry
        features['eye_size_symmetry'] = min(left_width, right_width) / (max(left_width, right_width) + 1e-6)
        
        # Eye tilt angle
        eye_tilt = np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        )
        features['eye_tilt_angle'] = np.degrees(eye_tilt)
        
        # Approximate aspect ratios (assuming typical eye shape)
        features['left_eye_aspect_ratio'] = 0.35  # Default estimate
        features['right_eye_aspect_ratio'] = 0.35
        
        return features
    
    def extract_nose_features(self, landmarks, inter_eye_dist):
        """Extract nose-related geometric features."""
        features = {}
        
        nose_bridge = np.array(landmarks['nose_bridge'])
        nose_tip = np.array(landmarks['nose_tip'])
        left_nostril = np.array(landmarks['left_nostril'])
        right_nostril = np.array(landmarks['right_nostril'])
        
        # Nose bridge length
        bridge_length = self.euclidean_distance(nose_bridge, nose_tip)
        features['nose_bridge_length_ratio'] = bridge_length / (inter_eye_dist + 1e-6)
        
        # Nose width
        nose_width = self.euclidean_distance(left_nostril, right_nostril)
        features['nose_width_ratio'] = nose_width / (inter_eye_dist + 1e-6)
        features['nose_base_width_ratio'] = nose_width / (inter_eye_dist + 1e-6)
        
        # Nose symmetry
        nose_center = (nose_tip[0], nose_tip[1])
        left_dist = abs(left_nostril[0] - nose_center[0])
        right_dist = abs(right_nostril[0] - nose_center[0])
        features['nose_symmetry'] = min(left_dist, right_dist) / (max(left_dist, right_dist) + 1e-6)
        
        return features
    
    def extract_jaw_features(self, landmarks, inter_eye_dist):
        """Extract jaw and face shape features."""
        features = {}
        
        chin = np.array(landmarks['chin'])
        left_jaw = np.array(landmarks['left_jaw'])
        right_jaw = np.array(landmarks['right_jaw'])
        forehead = np.array(landmarks['forehead_center'])
        
        # Jaw width
        jaw_width = self.euclidean_distance(left_jaw, right_jaw)
        features['jaw_width_ratio'] = jaw_width / (inter_eye_dist + 1e-6)
        
        # Face height
        face_height = self.euclidean_distance(forehead, chin)
        features['face_height_ratio'] = face_height / (inter_eye_dist + 1e-6)
        
        # Face aspect ratio
        features['face_aspect_ratio'] = jaw_width / (face_height + 1e-6)
        
        # Chin angle
        chin_angle = self.angle_between_points(left_jaw, chin, right_jaw)
        features['chin_angle'] = chin_angle
        
        # Jaw angles
        left_cheek = np.array(landmarks['left_cheek'])
        right_cheek = np.array(landmarks['right_cheek'])
        
        features['left_jaw_angle'] = self.angle_between_points(left_cheek, left_jaw, chin)
        features['right_jaw_angle'] = self.angle_between_points(right_cheek, right_jaw, chin)
        
        # Jaw symmetry
        features['jaw_symmetry'] = min(features['left_jaw_angle'], features['right_jaw_angle']) / (
            max(features['left_jaw_angle'], features['right_jaw_angle']) + 1e-6
        )
        
        return features
    
    def extract_cheekbone_features(self, landmarks, inter_eye_dist):
        """Extract cheekbone features."""
        features = {}
        
        left_cheek = np.array(landmarks['left_cheek'])
        right_cheek = np.array(landmarks['right_cheek'])
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        left_jaw = np.array(landmarks['left_jaw'])
        right_jaw = np.array(landmarks['right_jaw'])
        
        # Cheekbone width
        cheekbone_width = self.euclidean_distance(left_cheek, right_cheek)
        features['cheekbone_width_ratio'] = cheekbone_width / (inter_eye_dist + 1e-6)
        
        # Cheekbone angles
        features['left_cheekbone_angle'] = self.angle_between_points(left_eye, left_cheek, left_jaw)
        features['right_cheekbone_angle'] = self.angle_between_points(right_eye, right_cheek, right_jaw)
        
        # Cheekbone symmetry
        features['cheekbone_symmetry'] = min(
            features['left_cheekbone_angle'], 
            features['right_cheekbone_angle']
        ) / (max(features['left_cheekbone_angle'], features['right_cheekbone_angle']) + 1e-6)
        
        return features
    
    def extract_mouth_features(self, landmarks, inter_eye_dist):
        """Extract mouth features."""
        features = {}
        
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        mouth_top = np.array(landmarks['mouth_top'])
        mouth_bottom = np.array(landmarks['mouth_bottom'])
        nose_tip = np.array(landmarks['nose_tip'])
        
        # Mouth width
        mouth_width = self.euclidean_distance(mouth_left, mouth_right)
        features['mouth_width_ratio'] = mouth_width / (inter_eye_dist + 1e-6)
        
        # Lip height
        lip_height = self.euclidean_distance(mouth_top, mouth_bottom)
        features['upper_lip_ratio'] = (lip_height * 0.4) / (inter_eye_dist + 1e-6)  # Estimate
        features['lower_lip_ratio'] = (lip_height * 0.6) / (inter_eye_dist + 1e-6)  # Estimate
        features['lip_ratio'] = features['upper_lip_ratio'] / (features['lower_lip_ratio'] + 1e-6)
        
        # Mouth to nose distance
        mouth_center = ((mouth_left[0] + mouth_right[0])//2, mouth_top[1])
        mouth_nose_dist = self.euclidean_distance(mouth_center, nose_tip)
        features['mouth_nose_distance_ratio'] = mouth_nose_dist / (inter_eye_dist + 1e-6)
        
        return features
    
    def extract_proportion_features(self, landmarks, inter_eye_dist):
        """Extract facial proportion and symmetry features."""
        features = {}
        
        forehead = np.array(landmarks['forehead_center'])
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose_tip = np.array(landmarks['nose_tip'])
        chin = np.array(landmarks['chin'])
        
        eye_center = ((left_eye[0] + right_eye[0])//2, (left_eye[1] + right_eye[1])//2)
        
        # Face thirds (ideal ratio is 1:1:1)
        upper_third = abs(eye_center[1] - forehead[1])
        middle_third = abs(nose_tip[1] - eye_center[1])
        lower_third = abs(chin[1] - nose_tip[1])
        
        total_height = upper_third + middle_third + lower_third
        ideal_third = total_height / 3
        
        features['upper_third_deviation'] = abs(upper_third - ideal_third) / (ideal_third + 1e-6)
        features['middle_third_deviation'] = abs(middle_third - ideal_third) / (ideal_third + 1e-6)
        features['lower_third_deviation'] = abs(lower_third - ideal_third) / (ideal_third + 1e-6)
        
        # Horizontal symmetry (compare left and right sides)
        left_cheek = np.array(landmarks['left_cheek'])
        right_cheek = np.array(landmarks['right_cheek'])
        
        center_x = (left_eye[0] + right_eye[0]) // 2
        left_dist = abs(left_cheek[0] - center_x)
        right_dist = abs(right_cheek[0] - center_x)
        
        features['horizontal_symmetry'] = min(left_dist, right_dist) / (max(left_dist, right_dist) + 1e-6)
        
        return features
    
    def extract_texture_features(self, face_roi):
        """
        Extract texture features using Local Binary Patterns (LBP).
        This captures unique skin texture patterns.
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # Resize to standard size for consistent features
        face_resized = cv2.resize(gray, (128, 128))
        
        # Simple LBP implementation
        lbp = np.zeros_like(face_resized, dtype=np.uint8)
        
        for i in range(1, face_resized.shape[0] - 1):
            for j in range(1, face_resized.shape[1] - 1):
                center = face_resized[i, j]
                code = 0
                code |= (face_resized[i-1, j-1] >= center) << 7
                code |= (face_resized[i-1, j] >= center) << 6
                code |= (face_resized[i-1, j+1] >= center) << 5
                code |= (face_resized[i, j+1] >= center) << 4
                code |= (face_resized[i+1, j+1] >= center) << 3
                code |= (face_resized[i+1, j] >= center) << 2
                code |= (face_resized[i+1, j-1] >= center) << 1
                code |= (face_resized[i, j-1] >= center) << 0
                lbp[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        
        # Reduce to key statistics
        texture_features = {
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp)),
            'lbp_entropy': float(-np.sum(hist * np.log2(hist + 1e-6))),
        }
        
        return texture_features, hist[:64]  # Return first 64 bins for vector
    
    def extract_all_features(self, image, face_box=None):
        """
        Extract all geometric and texture features from a face.
        
        Args:
            image: BGR image
            face_box: (x, y, w, h) tuple or None for auto-detect
            
        Returns:
            features_dict: Dictionary of named features
            feature_vector: Numpy array for similarity comparison
        """
        try:
            # Get face ROI
            if face_box is not None:
                x, y, w, h = [int(v) for v in face_box]
                # Add padding
                pad = int(min(w, h) * 0.1)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                face_roi = image[y1:y2, x1:x2]
                local_box = (0, 0, face_roi.shape[1], face_roi.shape[0])
            else:
                # Detect face
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) == 0:
                    return None, None
                x, y, w, h = faces[0]
                face_roi = image[y:y+h, x:x+w]
                local_box = (0, 0, w, h)
            
            if face_roi.size == 0:
                return None, None
            
            # Estimate landmarks
            landmarks, inter_eye_dist, eyes_detected = self.estimate_landmarks(face_roi, local_box)
            
            if inter_eye_dist < 10:  # Face too small
                return None, None
            
            # Extract all feature categories
            features = {}
            
            eye_features = self.extract_eye_features(landmarks, inter_eye_dist)
            features.update(eye_features)
            
            nose_features = self.extract_nose_features(landmarks, inter_eye_dist)
            features.update(nose_features)
            
            jaw_features = self.extract_jaw_features(landmarks, inter_eye_dist)
            features.update(jaw_features)
            
            cheekbone_features = self.extract_cheekbone_features(landmarks, inter_eye_dist)
            features.update(cheekbone_features)
            
            mouth_features = self.extract_mouth_features(landmarks, inter_eye_dist)
            features.update(mouth_features)
            
            proportion_features = self.extract_proportion_features(landmarks, inter_eye_dist)
            features.update(proportion_features)
            
            texture_features, lbp_hist = self.extract_texture_features(face_roi)
            features.update(texture_features)
            
            # Quality indicator
            features['eyes_detected'] = 1.0 if eyes_detected else 0.0
            
            # Build feature vector (normalized, fixed length)
            feature_vector = np.array([
                # Eye features (7)
                features['inter_eye_distance'],
                features['left_eye_width_ratio'],
                features['right_eye_width_ratio'],
                features['left_eye_aspect_ratio'],
                features['right_eye_aspect_ratio'],
                features['eye_size_symmetry'],
                features['eye_tilt_angle'] / 45.0,  # Normalize
                
                # Nose features (4)
                features['nose_bridge_length_ratio'],
                features['nose_width_ratio'],
                features['nose_base_width_ratio'],
                features['nose_symmetry'],
                
                # Jaw features (7)
                features['jaw_width_ratio'],
                features['face_height_ratio'],
                features['face_aspect_ratio'],
                features['chin_angle'] / 180.0,  # Normalize
                features['left_jaw_angle'] / 180.0,
                features['right_jaw_angle'] / 180.0,
                features['jaw_symmetry'],
                
                # Cheekbone features (4)
                features['cheekbone_width_ratio'],
                features['left_cheekbone_angle'] / 180.0,
                features['right_cheekbone_angle'] / 180.0,
                features['cheekbone_symmetry'],
                
                # Mouth features (5)
                features['mouth_width_ratio'],
                features['upper_lip_ratio'],
                features['lower_lip_ratio'],
                features['lip_ratio'],
                features['mouth_nose_distance_ratio'],
                
                # Proportion features (4)
                features['upper_third_deviation'],
                features['middle_third_deviation'],
                features['lower_third_deviation'],
                features['horizontal_symmetry'],
                
                # Texture features (3)
                features['lbp_mean'] / 255.0,
                features['lbp_std'] / 128.0,
                features['lbp_entropy'] / 8.0,
            ], dtype=np.float32)
            
            # Append LBP histogram (reduced)
            feature_vector = np.concatenate([feature_vector, lbp_hist[:32].astype(np.float32)])
            
            return features, feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def compare_features(self, features1, features2):
        """
        Compare two feature vectors using cosine similarity.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            float: Similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure same length
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Normalize
        f1 = f1 / (np.linalg.norm(f1) + 1e-6)
        f2 = f2 / (np.linalg.norm(f2) + 1e-6)
        
        # Cosine similarity
        similarity = np.dot(f1, f2)
        
        return float(np.clip(similarity, 0, 1))


class HybridFaceRecognizer:
    """
    Combines deep learning embeddings with geometric feature analysis.
    Uses weighted combination of:
    - SFace/ArcFace deep embeddings (primary)
    - Geometric facial measurements (secondary)
    """
    
    def __init__(self, sface_model_path="face_recognition_sface_2021dec.onnx"):
        """Initialize hybrid recognizer."""
        # Deep embedding model (SFace)
        self.sface = cv2.FaceRecognizerSF.create(
            model=sface_model_path,
            config="",
            backend_id=0,
            target_id=0
        )
        print("✅ SFace deep embedding model loaded")
        
        # Geometric feature extractor
        self.geometric = AdvancedFaceFeatures()
        print("✅ Geometric feature extractor initialized")
        
        # Weights for combining scores
        self.deep_weight = 0.7  # 70% deep embeddings
        self.geometric_weight = 0.3  # 30% geometric features
        
    def get_embedding(self, image, face_box):
        """
        Get combined embedding from deep model and geometric features.
        
        Args:
            image: BGR image
            face_box: (x, y, w, h) face bounding box
            
        Returns:
            dict: Contains 'deep_embedding', 'geometric_features', 'geometric_vector'
        """
        x, y, w, h = face_box
        face_box_np = np.array([x, y, w, h], dtype=np.int32)
        
        result = {
            'deep_embedding': None,
            'geometric_features': None,
            'geometric_vector': None
        }
        
        try:
            # Get deep embedding (SFace)
            aligned = self.sface.alignCrop(image, face_box_np)
            result['deep_embedding'] = self.sface.feature(aligned)
        except Exception as e:
            print(f"Deep embedding error: {e}")
            
        try:
            # Get geometric features
            features_dict, feature_vector = self.geometric.extract_all_features(image, face_box)
            result['geometric_features'] = features_dict
            result['geometric_vector'] = feature_vector
        except Exception as e:
            print(f"Geometric features error: {e}")
            
        return result
    
    def compare(self, embedding1, embedding2):
        """
        Compare two embeddings using weighted combination.
        
        Returns:
            float: Combined similarity score (0-1)
            dict: Individual scores
        """
        scores = {
            'deep_score': 0.0,
            'geometric_score': 0.0,
            'combined_score': 0.0
        }
        
        # Deep embedding comparison
        if embedding1['deep_embedding'] is not None and embedding2['deep_embedding'] is not None:
            scores['deep_score'] = float(self.sface.match(
                embedding1['deep_embedding'],
                embedding2['deep_embedding'],
                cv2.FaceRecognizerSF_FR_COSINE
            ))
        
        # Geometric comparison
        if embedding1['geometric_vector'] is not None and embedding2['geometric_vector'] is not None:
            scores['geometric_score'] = self.geometric.compare_features(
                embedding1['geometric_vector'],
                embedding2['geometric_vector']
            )
        
        # Combined score (weighted)
        if scores['deep_score'] > 0 and scores['geometric_score'] > 0:
            scores['combined_score'] = (
                self.deep_weight * scores['deep_score'] + 
                self.geometric_weight * scores['geometric_score']
            )
        elif scores['deep_score'] > 0:
            scores['combined_score'] = scores['deep_score']
        else:
            scores['combined_score'] = scores['geometric_score']
            
        return scores['combined_score'], scores


# Test the module
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Advanced Face Feature Extraction Test (OpenCV)")
    print("="*60 + "\n")
    
    # Test with a sample image
    import sys
    
    if len(sys.argv) > 1:
        test_image = cv2.imread(sys.argv[1])
        if test_image is not None:
            extractor = AdvancedFaceFeatures()
            features, vector = extractor.extract_all_features(test_image)
            
            if features:
                print("Extracted Features:")
                print("-" * 40)
                for name, value in features.items():
                    if isinstance(value, float):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: {value}")
                print("-" * 40)
                if vector is not None:
                    print(f"Feature vector shape: {vector.shape}")
            else:
                print("❌ Could not extract features (no face detected)")
        else:
            print(f"❌ Could not load image: {sys.argv[1]}")
    else:
        print("Usage: python advanced_face_features.py <image_path>")
        print("\nThis module uses OpenCV only (no dlib/mediapipe required)")
        print("Eye detection is used to estimate facial landmarks")
