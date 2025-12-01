"""
Automatic Registration Pipeline - 5-Photo Multi-Angle
Handles complete training data setup for new members
Creates full folder structure like aziz/ala/aidh with 5 angles
"""
import sys
import os
import shutil
from pathlib import Path
from train_person import train_person
import subprocess

def setup_new_member(member_id, first_name, last_name, photo_paths):
    """
    Complete registration pipeline for a new member with 5-angle photos.
    
    Args:
        member_id: Member database ID
        first_name: First name
        last_name: Last name  
        photo_paths: Dict with keys: 'center', 'left', 'right', 'up', 'down'
                     OR list of paths in order [center, left, right, up, down]
    
    Returns:
        dict with status and paths
    """
    print("=" * 70)
    print(f"🎬 Starting 5-Photo Registration Pipeline for {first_name} {last_name}")
    print("=" * 70)
    
    # Handle both dict and list formats
    if isinstance(photo_paths, dict):
        photos = photo_paths
    else:
        # Assume list in order: center, left, right, up, down
        angle_names = ['center', 'left', 'right', 'up', 'down']
        photos = {angle_names[i]: photo_paths[i] for i in range(min(len(photo_paths), 5))}
    
    # Create person folder structure
    name_normalized = f"{first_name}_{last_name}".replace(" ", "_")
    folder_name = f"{member_id}_{name_normalized}"
    base_path = Path("training_data")
    person_path = base_path / folder_name
    
    # Create all required folders (like aziz/ala/aidh structure)
    raw_path = person_path / "raw"
    cropped_path = person_path / "cropped"
    augmented_path = person_path / "augmented"
    embeddings_path = person_path / "embeddings"
    features_path = person_path / "features"
    
    print(f"\n📁 Creating folder structure...")
    for folder in [raw_path, cropped_path, augmented_path, embeddings_path, features_path]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {folder}")
    
    # Copy all 5 photos to raw folder
    print(f"\n📸 Copying 5-angle photos to raw folder...")
    copied_count = 0
    for angle, photo_path in photos.items():
        dest_photo = raw_path / f"{angle}.jpg"
        try:
            if os.path.exists(photo_path):
                shutil.copy2(photo_path, dest_photo)
                print(f"   ✅ {angle}.jpg copied")
                copied_count += 1
            else:
                print(f"   ⚠️ {angle}.jpg - source not found: {photo_path}")
        except Exception as e:
            print(f"   ❌ Failed to copy {angle}.jpg: {e}")
    
    if copied_count == 0:
        print(f"\n❌ No photos were copied!")
        return {
            'success': False,
            'error': 'Failed to copy any photos'
        }
    
    print(f"\n✅ Copied {copied_count}/5 photos to raw folder")
    
    # Run training pipeline (cropping + augmentation)
    print(f"\n🎨 Starting data augmentation pipeline...")
    try:
        result = train_person(
            person_id=member_id,
            name=f"{first_name} {last_name}",
            num_augmentations=37,  # 37 augmentations = ~190 total images
            verbose=True
        )
        
        if result:
            print(f"\n✅ Augmentation complete: {result['augmented_images']} images generated")
        else:
            print(f"\n⚠️ Augmentation failed or no images found")
            return {
                'success': False,
                'error': 'Augmentation pipeline failed'
            }
    except Exception as e:
        print(f"\n❌ Augmentation error: {e}")
        return {
            'success': False,
            'error': f'Augmentation failed: {e}'
        }
    
    # Update ArcFace database via API (incremental, fast)
    print(f"\n🔄 Updating ArcFace recognition database...")
    try:
        import requests
        import base64
        
        # Get the center image to register with ArcFace
        center_img_path = cropped_path / "center.jpg"
        if not center_img_path.exists():
            # Try any cropped image
            cropped_images = list(cropped_path.glob('*.jpg'))
            if cropped_images:
                center_img_path = cropped_images[0]
        
        if center_img_path.exists():
            # Encode image to base64
            with open(center_img_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Call ArcFace API /register endpoint
            response = requests.post(
                'http://localhost:5001/register',
                json={
                    'member_id': member_id,
                    'name': f"{first_name} {last_name}",
                    'image': img_b64
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ ArcFace database updated incrementally")
                    print(f"   Embeddings count: {result.get('embeddings_count', 1)}")
                else:
                    print(f"⚠️ API returned error: {result.get('error', 'Unknown')}")
            else:
                print(f"⚠️ API returned status {response.status_code}")
        else:
            print(f"⚠️ No cropped image found, skipping ArcFace update")
            print(f"   Run 'python build_arcface_db.py' manually to rebuild database")
            
    except requests.exceptions.ConnectionError:
        print(f"⚠️ ArcFace API not running. Start it with: python arcface_api.py")
        print(f"   Or run 'python build_arcface_db.py' to rebuild database offline")
    except Exception as e:
        print(f"⚠️ ArcFace update error: {e}")
        print(f"   You can rebuild the full database with: python build_arcface_db.py")
    
    print("\n" + "=" * 70)
    print(f"✅ Registration Pipeline Complete!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   - Folder: {person_path}")
    print(f"   - Raw images: {len(list(raw_path.glob('*.jpg')))}")
    print(f"   - Cropped images: {len(list(cropped_path.glob('*.jpg')))}")
    print(f"   - Augmented images: {len(list(augmented_path.glob('*.jpg')))}")
    print(f"\n💡 Member can now be recognized in the scanner!")
    
    return {
        'success': True,
        'folder_path': str(person_path),
        'raw_images': len(list(raw_path.glob('*.jpg'))),
        'augmented_images': len(list(augmented_path.glob('*.jpg')))
    }


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python register_member.py <member_id> <first_name> <last_name> <center_photo> <left_photo> <right_photo> <up_photo> <down_photo>")
        print("\nOr provide at least center photo:")
        print("Usage: python register_member.py <member_id> <first_name> <last_name> <center_photo>")
        sys.exit(1)
    
    member_id = int(sys.argv[1])
    first_name = sys.argv[2]
    last_name = sys.argv[3]
    
    # Collect photo paths
    photos = {}
    angle_names = ['center', 'left', 'right', 'up', 'down']
    for i, angle in enumerate(angle_names):
        if len(sys.argv) > 4 + i:
            photos[angle] = sys.argv[4 + i]
    
    if not photos:
        print("❌ At least one photo is required")
        sys.exit(1)
    
    print(f"📸 Registering with {len(photos)} photo(s): {', '.join(photos.keys())}")
    
    result = setup_new_member(member_id, first_name, last_name, photos)
    
    if result['success']:
        print(f"\n✅ SUCCESS")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)
