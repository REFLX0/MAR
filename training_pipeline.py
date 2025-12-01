"""
Advanced Training Data Pipeline for YOLO Face Recognition
=========================================================
Processes training_data folder with:
- Raw images processing
- Face detection and cropping
- Data augmentation (rotation, flip, brightness, etc.)
- Feature extraction (geometric + deep embeddings)
- Organized output structure for YOLO training

Structure per person:
training_data/
└── person_name/
    ├── raw/           (original images)
    ├── cropped/       (detected & cropped faces)
    ├── augmented/     (augmented versions)
    ├── features/      (extracted feature files)
    └── embeddings/    (deep face embeddings)
"""

import os
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from advanced_face_features import AdvancedFaceFeatures
    FEATURES_AVAILABLE = True
    print("✅ Advanced face features loaded")
except ImportError as e:
    FEATURES_AVAILABLE = False
    print(f"⚠️ Advanced face features not available: {e}")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available, using OpenCV cascade")


class FaceDetector:
    """Multi-backend face detector with YOLO priority"""
    
    def __init__(self):
        self.detector = None
        self.backend = None
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the best available detector"""
        # Try YOLO first
        if YOLO_AVAILABLE:
            try:
                model_path = Path(__file__).parent / "yolov8n.pt"
                if model_path.exists():
                    self.detector = YOLO(str(model_path))
                    self.backend = "yolo"
                    print("✅ Using YOLO face detector")
                    return
            except Exception as e:
                print(f"⚠️ YOLO init failed: {e}")
        
        # Fallback to OpenCV DNN
        try:
            prototxt = cv2.data.haarcascades + "../dnn/deploy.prototxt"
            caffemodel = cv2.data.haarcascades + "../dnn/res10_300x300_ssd_iter_140000.caffemodel"
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.backend = "dnn"
                print("✅ Using OpenCV DNN face detector")
                return
        except:
            pass
        
        # Final fallback: Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.backend = "haar"
        print("✅ Using Haar Cascade face detector")
    
    def detect(self, image, conf_threshold=0.5):
        """Detect faces and return bounding boxes [(x, y, w, h, confidence), ...]"""
        if image is None:
            return []
        
        h, w = image.shape[:2]
        faces = []
        
        if self.backend == "yolo":
            results = self.detector(image, verbose=False)
            for result in results:
                for box in result.boxes:
                    # YOLO returns person class, filter for faces based on size
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    bw, bh = x2 - x1, y2 - y1
                    # Check if detection looks like a face (roughly square, reasonable size)
                    if conf >= conf_threshold and 0.5 < bw/max(bh,1) < 2.0:
                        faces.append((x1, y1, bw, bh, conf))
        
        elif self.backend == "dnn":
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf >= conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2-x1, y2-y1, conf))
        
        else:  # haar
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            detections = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, bw, bh) in detections:
                faces.append((x, y, bw, bh, 1.0))
        
        return faces


class DataAugmenter:
    """Advanced data augmentation for face images"""
    
    def __init__(self):
        self.augmentations = [
            ("original", lambda img: img),
            ("flip_h", self._flip_horizontal),
            ("rotate_5", lambda img: self._rotate(img, 5)),
            ("rotate_-5", lambda img: self._rotate(img, -5)),
            ("rotate_10", lambda img: self._rotate(img, 10)),
            ("rotate_-10", lambda img: self._rotate(img, -10)),
            ("bright_up", lambda img: self._adjust_brightness(img, 1.2)),
            ("bright_down", lambda img: self._adjust_brightness(img, 0.8)),
            ("contrast_up", lambda img: self._adjust_contrast(img, 1.3)),
            ("contrast_down", lambda img: self._adjust_contrast(img, 0.7)),
            ("blur_light", lambda img: cv2.GaussianBlur(img, (3, 3), 0)),
            ("sharpen", self._sharpen),
            ("noise", self._add_noise),
            ("scale_up", lambda img: self._scale(img, 1.1)),
            ("scale_down", lambda img: self._scale(img, 0.9)),
            ("gamma_up", lambda img: self._adjust_gamma(img, 1.2)),
            ("gamma_down", lambda img: self._adjust_gamma(img, 0.8)),
        ]
    
    def _flip_horizontal(self, image):
        return cv2.flip(image, 1)
    
    def _rotate(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def _adjust_brightness(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _adjust_contrast(self, image, factor):
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _sharpen(self, image):
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _add_noise(self, image, noise_level=10):
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def _scale(self, image, factor):
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        scaled = cv2.resize(image, (new_w, new_h))
        # Crop or pad to original size
        if factor > 1:
            start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
            return scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            result = np.zeros_like(image)
            start_y, start_x = (h - new_h) // 2, (w - new_w) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
            return result
    
    def _adjust_gamma(self, image, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def augment(self, image, augmentation_names=None):
        """Generate augmented versions of an image
        
        Args:
            image: Input image
            augmentation_names: List of specific augmentations to apply, or None for all
            
        Returns:
            List of (name, augmented_image) tuples
        """
        results = []
        
        for name, func in self.augmentations:
            if augmentation_names is None or name in augmentation_names:
                try:
                    aug_img = func(image.copy())
                    if aug_img is not None and aug_img.shape == image.shape:
                        results.append((name, aug_img))
                except Exception as e:
                    print(f"⚠️ Augmentation '{name}' failed: {e}")
        
        return results


class TrainingDataPipeline:
    """Main pipeline for processing training data"""
    
    def __init__(self, training_dir="training_data", output_size=(224, 224)):
        self.training_dir = Path(training_dir)
        self.output_size = output_size
        self.detector = FaceDetector()
        self.augmenter = DataAugmenter()
        
        # Initialize feature extractor if available
        self.feature_extractor = None
        if FEATURES_AVAILABLE:
            try:
                self.feature_extractor = AdvancedFaceFeatures()
                print("✅ Feature extractor initialized")
            except Exception as e:
                print(f"⚠️ Feature extractor init failed: {e}")
        
        # Load SFace for embeddings
        self.sface = None
        sface_path = Path(__file__).parent / "face_recognition_sface_2021dec.onnx"
        if sface_path.exists():
            try:
                self.sface = cv2.FaceRecognizerSF.create(str(sface_path), "")
                print("✅ SFace embeddings initialized")
            except Exception as e:
                print(f"⚠️ SFace init failed: {e}")
        
        self.stats = {
            "persons_processed": 0,
            "images_processed": 0,
            "faces_detected": 0,
            "augmentations_created": 0,
            "features_extracted": 0,
            "embeddings_created": 0,
            "errors": []
        }
    
    def get_person_dirs(self):
        """Get all person directories in training folder"""
        persons = []
        if self.training_dir.exists():
            for item in sorted(self.training_dir.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    persons.append(item)
        return persons
    
    def ensure_person_structure(self, person_dir):
        """Create complete folder structure for a person"""
        folders = ["raw", "cropped", "augmented", "features", "embeddings"]
        for folder in folders:
            (person_dir / folder).mkdir(parents=True, exist_ok=True)
        return {f: person_dir / f for f in folders}
    
    def crop_face(self, image, bbox, padding=0.3):
        """Crop face from image with padding"""
        x, y, w, h = bbox[:4]
        img_h, img_w = image.shape[:2]
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        face = image[y1:y2, x1:x2]
        
        # Resize to standard size
        if face.size > 0:
            face = cv2.resize(face, self.output_size)
        
        return face
    
    def extract_embedding(self, face_image):
        """Extract deep face embedding using SFace"""
        if self.sface is None:
            return None
        
        try:
            # Align face for better embedding
            aligned = cv2.resize(face_image, (112, 112))
            embedding = self.sface.feature(aligned)
            return embedding.flatten()
        except Exception as e:
            return None
    
    def process_person(self, person_dir, force_reprocess=False):
        """Process all data for a single person"""
        person_name = person_dir.name
        print(f"\n{'='*60}")
        print(f"📁 Processing: {person_name}")
        print(f"{'='*60}")
        
        # Ensure folder structure
        folders = self.ensure_person_structure(person_dir)
        
        # Collect all raw images
        raw_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            raw_images.extend(folders["raw"].glob(ext))
            raw_images.extend(folders["raw"].glob(ext.upper()))
        
        if not raw_images:
            print(f"⚠️ No raw images found in {folders['raw']}")
            return
        
        print(f"📸 Found {len(raw_images)} raw images")
        
        person_embeddings = []
        person_features = []
        
        for img_path in sorted(raw_images):
            img_name = img_path.stem
            print(f"\n  Processing: {img_name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    ❌ Failed to load image")
                self.stats["errors"].append(f"Failed to load: {img_path}")
                continue
            
            self.stats["images_processed"] += 1
            
            # Detect faces
            faces = self.detector.detect(image)
            if not faces:
                print(f"    ⚠️ No face detected, using full image")
                # Use center crop as fallback
                h, w = image.shape[:2]
                min_dim = min(h, w)
                x = (w - min_dim) // 2
                y = (h - min_dim) // 2
                faces = [(x, y, min_dim, min_dim, 0.5)]
            
            print(f"    ✅ Detected {len(faces)} face(s)")
            self.stats["faces_detected"] += len(faces)
            
            # Process best face (highest confidence)
            best_face = max(faces, key=lambda f: f[4])
            cropped = self.crop_face(image, best_face)
            
            if cropped is None or cropped.size == 0:
                print(f"    ❌ Failed to crop face")
                continue
            
            # Save cropped face
            cropped_path = folders["cropped"] / f"{img_name}_cropped.jpg"
            cv2.imwrite(str(cropped_path), cropped)
            print(f"    💾 Saved cropped face")
            
            # Generate augmentations
            augmentations = self.augmenter.augment(cropped)
            for aug_name, aug_img in augmentations:
                aug_path = folders["augmented"] / f"{img_name}_{aug_name}.jpg"
                cv2.imwrite(str(aug_path), aug_img)
                self.stats["augmentations_created"] += 1
                
                # Extract embedding for each augmentation
                embedding = self.extract_embedding(aug_img)
                if embedding is not None:
                    person_embeddings.append({
                        "source": img_name,
                        "augmentation": aug_name,
                        "embedding": embedding.tolist()
                    })
                    self.stats["embeddings_created"] += 1
            
            print(f"    🔄 Created {len(augmentations)} augmentations")
            
            # Extract geometric features from original cropped face
            if self.feature_extractor:
                try:
                    result = self.feature_extractor.extract_all_features(cropped)
                    if result is not None and result[0] is not None:
                        features_dict, feature_vector = result
                        features_dict["source_image"] = img_name
                        features_dict["feature_vector"] = feature_vector.tolist() if feature_vector is not None else None
                        person_features.append(features_dict)
                        self.stats["features_extracted"] += 1
                        print(f"    📊 Extracted {len(features_dict)} features")
                except Exception as e:
                    print(f"    ⚠️ Feature extraction failed: {e}")
        
        # Save all embeddings for this person
        if person_embeddings:
            embeddings_file = folders["embeddings"] / "embeddings.json"
            with open(embeddings_file, 'w') as f:
                json.dump({
                    "person_id": person_name,
                    "created_at": datetime.now().isoformat(),
                    "count": len(person_embeddings),
                    "embeddings": person_embeddings
                }, f, indent=2)
            
            # Also save as numpy for fast loading
            embeddings_np = np.array([e["embedding"] for e in person_embeddings])
            np.save(folders["embeddings"] / "embeddings.npy", embeddings_np)
            print(f"\n  💾 Saved {len(person_embeddings)} embeddings")
        
        # Save geometric features
        if person_features:
            features_file = folders["features"] / "geometric_features.json"
            with open(features_file, 'w') as f:
                json.dump({
                    "person_id": person_name,
                    "created_at": datetime.now().isoformat(),
                    "features": person_features
                }, f, indent=2)
            
            # Create feature summary (averages)
            self._create_feature_summary(person_features, folders["features"])
            print(f"  📊 Saved geometric features")
        
        self.stats["persons_processed"] += 1
        print(f"\n✅ Completed processing: {person_name}")
    
    def _create_feature_summary(self, features_list, output_dir):
        """Create averaged feature summary for a person"""
        if not features_list:
            return
        
        # Collect numeric features (flat structure)
        numeric_features = {}
        feature_vectors = []
        
        for features in features_list:
            for key, val in features.items():
                if key == 'source_image':
                    continue
                if key == 'feature_vector' and val is not None:
                    feature_vectors.append(val)
                    continue
                if isinstance(val, (int, float)) and not np.isnan(val):
                    if key not in numeric_features:
                        numeric_features[key] = []
                    numeric_features[key].append(val)
        
        # Calculate statistics
        summary = {}
        for key, values in numeric_features.items():
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
        
        # Save summary
        summary_file = output_dir / "feature_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create averaged feature vector from extracted vectors
        if feature_vectors:
            avg_vector = np.mean(feature_vectors, axis=0)
            np.save(output_dir / "feature_vector.npy", avg_vector)
        else:
            # Fallback: create from summary means
            feature_vector = np.array([v["mean"] for v in summary.values()])
            np.save(output_dir / "feature_vector.npy", feature_vector)
    
    def process_all(self, force_reprocess=False):
        """Process all persons in training directory"""
        print("\n" + "="*70)
        print("🚀 TRAINING DATA PIPELINE - Starting Processing")
        print("="*70)
        print(f"📂 Training directory: {self.training_dir}")
        print(f"📐 Output size: {self.output_size}")
        print(f"🔄 Force reprocess: {force_reprocess}")
        
        persons = self.get_person_dirs()
        print(f"👥 Found {len(persons)} persons to process")
        
        if not persons:
            print("❌ No person directories found!")
            return
        
        for person_dir in persons:
            try:
                self.process_person(person_dir, force_reprocess)
            except Exception as e:
                print(f"❌ Error processing {person_dir.name}: {e}")
                self.stats["errors"].append(f"{person_dir.name}: {str(e)}")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"👥 Persons processed:     {self.stats['persons_processed']}")
        print(f"📸 Images processed:      {self.stats['images_processed']}")
        print(f"👤 Faces detected:        {self.stats['faces_detected']}")
        print(f"🔄 Augmentations created: {self.stats['augmentations_created']}")
        print(f"📊 Features extracted:    {self.stats['features_extracted']}")
        print(f"🧠 Embeddings created:    {self.stats['embeddings_created']}")
        
        if self.stats["errors"]:
            print(f"\n⚠️ Errors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:10]:
                print(f"   - {error}")
        
        print("="*70)
    
    def build_recognition_database(self, output_file="recognition_database.pkl"):
        """Build unified recognition database from all processed data"""
        print("\n🔨 Building recognition database...")
        
        database = {
            "created_at": datetime.now().isoformat(),
            "persons": {},
            "embeddings_dim": None,
            "features_dim": None
        }
        
        for person_dir in self.get_person_dirs():
            person_name = person_dir.name
            folders = self.ensure_person_structure(person_dir)
            
            person_data = {
                "name": person_name,
                "embeddings": [],
                "features": None,
                "feature_vector": None
            }
            
            # Load embeddings
            embeddings_file = folders["embeddings"] / "embeddings.npy"
            if embeddings_file.exists():
                embeddings = np.load(embeddings_file)
                person_data["embeddings"] = embeddings
                if database["embeddings_dim"] is None:
                    database["embeddings_dim"] = embeddings.shape[1]
            
            # Load feature vector
            feature_file = folders["features"] / "feature_vector.npy"
            if feature_file.exists():
                features = np.load(feature_file)
                person_data["feature_vector"] = features
                if database["features_dim"] is None:
                    database["features_dim"] = len(features)
            
            # Load feature summary
            summary_file = folders["features"] / "feature_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    person_data["features"] = json.load(f)
            
            database["persons"][person_name] = person_data
        
        # Save database
        output_path = self.training_dir.parent / output_file
        with open(output_path, 'wb') as f:
            pickle.dump(database, f)
        
        print(f"✅ Database saved to: {output_path}")
        print(f"   Persons: {len(database['persons'])}")
        print(f"   Embedding dim: {database['embeddings_dim']}")
        print(f"   Features dim: {database['features_dim']}")
        
        return database


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Data Pipeline")
    parser.add_argument("--training-dir", default="training_data", help="Training data directory")
    parser.add_argument("--output-size", type=int, default=224, help="Output image size")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    parser.add_argument("--build-db", action="store_true", help="Build recognition database after processing")
    
    args = parser.parse_args()
    
    pipeline = TrainingDataPipeline(
        training_dir=args.training_dir,
        output_size=(args.output_size, args.output_size)
    )
    
    pipeline.process_all(force_reprocess=args.force)
    
    if args.build_db:
        pipeline.build_recognition_database()


if __name__ == "__main__":
    main()
