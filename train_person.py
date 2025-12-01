"""
Automated Person Training Script
=================================
Trains face recognition for a specific person using data augmentation.

Usage:
    python train_person.py --id 1 --name "John Doe" --augmentations 10
    
Workflow:
    1. Reads raw images from training_data/{id}_{name}/raw/
    2. Generates {augmentations} variations per image
    3. Saves to training_data/{id}_{name}/augmented/
    4. Total: 5 raw + (5 × augmentations) images
"""

import os
import sys
import argparse
from pathlib import Path
from augment_images import ImageAugmentor, augment_dataset
import cv2


def train_person(person_id, name, num_augmentations=10, verbose=True):
    """
    Generate augmented training data for a specific person.
    
    Args:
        person_id: Unique person ID
        name: Person's name (e.g., "John Doe")
        num_augmentations: Number of augmented versions per image
        verbose: Print progress
    
    Returns:
        dict with statistics
    """
    # Create folder name
    folder_name = f"{person_id}_{name.replace(' ', '_')}"
    base_path = Path("training_data")
    person_path = base_path / folder_name
    raw_path = person_path / "raw"
    augmented_path = person_path / "augmented"
    
    # Check if raw images exist
    if not raw_path.exists():
        print(f"❌ Error: Raw images folder not found: {raw_path}")
        print(f"   Please register person first using multi-angle registration")
        return None
    
    # Count raw images
    raw_images = list(raw_path.glob("*.jpg")) + list(raw_path.glob("*.png"))
    
    if len(raw_images) == 0:
        print(f"❌ Error: No images found in {raw_path}")
        return None
    
    if verbose:
        print("=" * 60)
        print(f"🎓 Training: {name} (ID: {person_id})")
        print("=" * 60)
        print(f"📁 Raw images: {len(raw_images)}")
    
    # 🚀 Auto-Crop with YOLOv11
    print("✂️  Auto-cropping faces with YOLOv11...")
    try:
        from face_detector_yolo import YOLOFaceDetector
        detector = YOLOFaceDetector(confidence_threshold=0.3)
        
        cropped_path = person_path / "cropped"
        cropped_path.mkdir(exist_ok=True)
        
        cropped_count = 0
        for img_file in raw_images:
            img = cv2.imread(str(img_file))
            detection = detector.detect_largest_face(img, return_crop=True)
            
            if detection:
                _, _, _, _, conf, crop = detection
                output_file = cropped_path / img_file.name
                cv2.imwrite(str(output_file), crop)
                print(f"  ✓ Cropped {img_file.name} (conf: {conf:.2f})")
                cropped_count += 1
            else:
                print(f"  ⚠️ No face found in {img_file.name}, using original")
                # Copy original if no face found
                output_file = cropped_path / img_file.name
                cv2.imwrite(str(output_file), img)
                cropped_count += 1
                
        # Use cropped images for augmentation
        input_dir = str(cropped_path)
        print(f"✅ Auto-cropped {cropped_count} images")
        
    except Exception as e:
        print(f"⚠️ Auto-crop failed: {e}")
        print("   Using raw images instead")
        input_dir = str(raw_path)

    if verbose:
        print(f"🔄 Augmentations per image: {num_augmentations}")
        print(f"📊 Total training images: {len(raw_images)} + {len(raw_images) * num_augmentations} = {len(raw_images) * (num_augmentations + 1)}")
        print("-" * 60)
    
    # Run augmentation
    augment_dataset(
        input_dir=input_dir,
        output_dir=str(augmented_path),
        augmentations_per_image=num_augmentations,
        verbose=verbose
    )
    
    # Count final images
    augmented_images = list(augmented_path.glob("*.jpg")) + list(augmented_path.glob("*.png"))
    
    stats = {
        'person_id': person_id,
        'name': name,
        'raw_images': len(raw_images),
        'augmented_images': len(augmented_images),
        'total_images': len(augmented_images),
        'folder_path': str(person_path)
    }
    
    if verbose:
        print("=" * 60)
        print(f"✅ Training complete!")
        print(f"📊 Generated {len(augmented_images)} total training images")
        print(f"💾 Saved to: {augmented_path}")
        print("=" * 60)
    
    return stats


def prepare_training_folders(person_id, name):
    """
    Create the folder structure for a new person.
    
    Args:
        person_id: Unique person ID
        name: Person's name
    
    Returns:
        Path to raw images folder
    """
    folder_name = f"{person_id}_{name.replace(' ', '_')}"
    base_path = Path("training_data")
    person_path = base_path / folder_name
    raw_path = person_path / "raw"
    augmented_path = person_path / "augmented"
    
    # Create folders
    raw_path.mkdir(parents=True, exist_ok=True)
    augmented_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created training folders for {name} (ID: {person_id})")
    print(f"   📁 {raw_path}")
    print(f"   📁 {augmented_path}")
    
    return raw_path


def list_trained_people():
    """List all people with training data."""
    base_path = Path("training_data")
    
    if not base_path.exists():
        print("❌ No training data found")
        return []
    
    people = []
    
    for person_folder in base_path.iterdir():
        if person_folder.is_dir() and "_" in person_folder.name:
            try:
                parts = person_folder.name.split("_", 1)
                person_id = int(parts[0])
                name = parts[1].replace("_", " ")
                
                raw_path = person_folder / "raw"
                augmented_path = person_folder / "augmented"
                
                raw_count = len(list(raw_path.glob("*.jpg"))) if raw_path.exists() else 0
                aug_count = len(list(augmented_path.glob("*.jpg"))) if augmented_path.exists() else 0
                
                people.append({
                    'id': person_id,
                    'name': name,
                    'raw_images': raw_count,
                    'augmented_images': aug_count,
                    'folder': str(person_folder)
                })
            except (ValueError, IndexError):
                continue
    
    return sorted(people, key=lambda p: p['id'])


def main():
    parser = argparse.ArgumentParser(
        description="Train face recognition for a specific person",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train person ID 1
  python train_person.py --id 1 --name "John Doe"
  
  # Train with 20 augmentations per image
  python train_person.py --id 1 --name "John Doe" -n 20
  
  # List all trained people
  python train_person.py --list
  
  # Create folders without training
  python train_person.py --id 1 --name "John Doe" --prepare
        """
    )
    
    parser.add_argument('--id', type=int, help='Person ID')
    parser.add_argument('--name', type=str, help='Person name')
    parser.add_argument('-n', '--augmentations', type=int, default=10,
                       help='Number of augmentations per image (default: 10)')
    parser.add_argument('--list', action='store_true',
                       help='List all trained people')
    parser.add_argument('--prepare', action='store_true',
                       help='Only create folders, don\'t train')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Trained People:")
        print("=" * 70)
        people = list_trained_people()
        
        if not people:
            print("No trained people found")
        else:
            for person in people:
                print(f"ID: {person['id']:3d} | {person['name']:20s} | "
                      f"Raw: {person['raw_images']:2d} | "
                      f"Aug: {person['augmented_images']:3d}")
        
        print("=" * 70)
        return
    
    if not args.id or not args.name:
        parser.print_help()
        print("\n❌ Error: --id and --name are required")
        sys.exit(1)
    
    if args.prepare:
        prepare_training_folders(args.id, args.name)
    else:
        result = train_person(
            person_id=args.id,
            name=args.name,
            num_augmentations=args.augmentations,
            verbose=not args.quiet
        )
        
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
