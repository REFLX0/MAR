"""
YOLOv11 Face Detector Module
============================
Fast and accurate face detection using YOLOv11.

Usage:
    from face_detector_yolo import YOLOFaceDetector
    
    detector = YOLOFaceDetector()
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h, confidence = face
        cropped_face = image[y:y+h, x:x+w]
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


class YOLOFaceDetector:
    """Face detection using YOLOv11."""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize YOLOv11 face detector.
        
        Args:
            model_path: Path to custom model, or None for default YOLOv11n
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        self.confidence_threshold = confidence_threshold
        
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Use YOLOv11n (nano) for speed, or 'yolov8n-face.pt' if available
            print("Loading YOLOv11n model...")
            self.model = YOLO('yolov8n.pt')  # Will auto-download
            
        print("✅ YOLOv11 face detector initialized")
    
    def detect_faces(self, image, return_crops=False):
        """
        Detect faces in an image.
        
        Args:
            image: numpy array (BGR format from cv2.imread)
            return_crops: If True, return cropped face images
            
        Returns:
            List of detections: [(x, y, w, h, confidence), ...]
            If return_crops=True: [(x, y, w, h, confidence, cropped_image), ...]
        """
        # Run YOLOv11 detection
        results = self.model(image, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Filter by confidence
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert to x, y, width, height
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                
                # Ensure valid coordinates
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if return_crops:
                    cropped = image[y:y+h, x:x+w]
                    detections.append((x, y, w, h, confidence, cropped))
                else:
                    detections.append((x, y, w, h, confidence))
        
        return detections
    
    def detect_largest_face(self, image, return_crop=True):
        """
        Detect and return only the largest face in the image.
        
        Args:
            image: numpy array (BGR format)
            return_crop: If True, return cropped face image
            
        Returns:
            If return_crop=True: (x, y, w, h, confidence, cropped_image)
            If return_crop=False: (x, y, w, h, confidence)
            None if no face detected
        """
        detections = self.detect_faces(image, return_crops=return_crop)
        
        if not detections:
            return None
        
        # Find largest face by area
        if return_crop:
            largest = max(detections, key=lambda d: d[2] * d[3])
        else:
            largest = max(detections, key=lambda d: d[2] * d[3])
        
        return largest
    
    def preprocess_for_recognition(self, face_crop, target_size=(160, 160)):
        """
        Preprocess face crop for recognition model.
        
        Args:
            face_crop: Cropped face image
            target_size: Output size (width, height)
            
        Returns:
            Preprocessed face image
        """
        # Resize to target size
        resized = cv2.resize(face_crop, target_size)
        
        # Optional: Histogram equalization for better lighting
        # lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        # lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        # resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return resized
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of (x, y, w, h, confidence) or with crops
            
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        
        for detection in detections:
            x, y, w, h, confidence = detection[:5]
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{confidence:.2f}"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output


def test_detector():
    """Test the face detector on a sample image."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python face_detector_yolo.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    
    # Initialize detector
    detector = YOLOFaceDetector(confidence_threshold=0.3)
    
    # Detect faces
    print(f"\nDetecting faces in: {image_path}")
    detections = detector.detect_faces(image, return_crops=True)
    
    print(f"Found {len(detections)} face(s)")
    
    # Draw and show results
    if detections:
        output = detector.draw_detections(image, detections)
        
        # Save result
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, output)
        print(f"✅ Saved result to: {output_path}")
        
        # Show individual face crops
        for i, detection in enumerate(detections):
            x, y, w, h, conf, crop = detection
            print(f"  Face {i+1}: {w}x{h}px, confidence: {conf:.3f}")
            cv2.imwrite(f"face_crop_{i+1}.jpg", crop)


if __name__ == "__main__":
    test_detector()
