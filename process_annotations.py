"""
Process Label Studio Face Annotations
=====================================
Reads Label Studio exports and crops faces for better training.

Usage:
    python process_annotations.py --input annotations.json --person-id 7
"""

import json
import cv2
import os
from pathlib import Path
import argparse


def process_label_studio_export(json_path, person_id, person_name):
    """
    Process Label Studio annotations and crop faces.
    
    Args:
        json_path: Path to Label Studio JSON export
        person_id: Member ID
        person_name: Person's name
    """
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output folder
    person_folder = f"{person_id}_{person_name.replace(' ', '_')}"
    raw_path = Path('training_data') / person_folder / 'raw'
    annotated_path = Path('training_data') / person_folder / 'annotated'
    annotated_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing annotations for {person_name} (ID: {person_id})")
    print(f"Output: {annotated_path}")
    print("-" * 60)
    
    processed = 0
    
    for item in data:
        # Get image path and annotations
        image_file = item.get('file_upload', item.get('image', ''))
        annotations = item.get('annotations', [])
        
        if not annotations:
            continue
        
        annotation = annotations[0]  # Use first annotation
        result = annotation.get('result', [])
        
        # Find face bounding box
        face_box = None
        quality = None
        angle = None
        
        for r in result:
            if r.get('type') == 'rectanglelabels':
                # Bounding box in percentage
                value = r.get('value', {})
                face_box = {
                    'x': value.get('x', 0),
                    'y': value.get('y', 0),
                    'width': value.get('width', 0),
                    'height': value.get('height', 0)
                }
            elif r.get('from_name') == 'quality':
                quality = r.get('value', {}).get('choices', [None])[0]
            elif r.get('from_name') == 'angle':
                angle = r.get('value', {}).get('choices', [None])[0]
        
        if not face_box:
            print(f"⚠️  No face box for {image_file}, skipping")
            continue
        
        # Load image
        image_name = Path(image_file).name
        image_path = raw_path / image_name
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Convert percentage to pixels
        x = int(face_box['x'] * w / 100)
        y = int(face_box['y'] * h / 100)
        box_w = int(face_box['width'] * w / 100)
        box_h = int(face_box['height'] * h / 100)
        
        # Crop face with padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + box_w + padding)
        y2 = min(h, y + box_h + padding)
        
        face_crop = img[y1:y2, x1:x2]
        
        # Save cropped face
        output_name = f"{angle.lower() if angle else image_name.stem}_cropped.jpg"
        output_path = annotated_path / output_name
        cv2.imwrite(str(output_path), face_crop)
        
        print(f"✓ {image_name} → {output_name} ({box_w}x{box_h}px, quality: {quality})")
        processed += 1
    
    print("-" * 60)
    print(f"✅ Processed {processed} images")
    print(f"💾 Saved to: {annotated_path}")
    
    # Save metadata
    metadata = {
        'person_id': person_id,
        'person_name': person_name,
        'processed_images': processed,
        'source': str(json_path)
    }
    
    with open(annotated_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return annotated_path


def main():
    parser = argparse.ArgumentParser(description="Process Label Studio face annotations")
    parser.add_argument('--input', '-i', required=True, help='Label Studio JSON export')
    parser.add_argument('--person-id', '-id', type=int, required=True, help='Person ID')
    parser.add_argument('--person-name', '-n', required=True, help='Person name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: File not found: {args.input}")
        return
    
    annotated_path = process_label_studio_export(
        args.input,
        args.person_id,
        args.person_name
    )
    
    # Ask if user wants to run augmentation
    print("\n" + "=" * 60)
    print("Next step: Run augmentation on cropped faces?")
    print(f"This will generate 55 training images from {annotated_path}")
    print("=" * 60)
    print("\nCommand:")
    print(f"python train_person.py --id {args.person_id} --name \"{args.person_name}\" -n 10")


if __name__ == '__main__':
    main()
