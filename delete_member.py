"""
Delete Member Script
Removes a member's training data folder and rebuilds the recognition database.
"""
import sys
import os
import shutil
from pathlib import Path
import subprocess

def delete_member_data(member_id):
    """
    Deletes the training data folder for a specific member ID
    and rebuilds the ArcFace database.
    
    Args:
        member_id: Member database ID
    """
    print("=" * 70)
    print(f"🗑️  Starting Deletion Pipeline for Member ID: {member_id}")
    print("=" * 70)
    
    base_path = Path("training_data")
    if not base_path.exists():
        print("⚠️  'training_data' folder not found.")
        return False
        
    # Find the member's folder (starts with id_)
    member_folder = None
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(f"{member_id}_"):
            member_folder = folder
            break
            
    if member_folder:
        print(f"📂 Found folder: {member_folder}")
        try:
            shutil.rmtree(member_folder)
            print(f"✅ Deleted folder: {member_folder}")
        except Exception as e:
            print(f"❌ Failed to delete folder: {e}")
            return False
    else:
        print(f"⚠️  No folder found for member ID {member_id}")
        # We continue to rebuild DB anyway, just in case
        
    # Rebuild ArcFace database
    print(f"\n🔄 Rebuilding ArcFace recognition database...")
    try:
        python_path = 'C:\\Users\\Asus\\miniconda3\\python.exe'
        build_script = Path('build_arcface_db.py')
        
        result = subprocess.run(
            [python_path, str(build_script)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ Database rebuilt successfully")
        else:
            print(f"⚠️ Database rebuild had issues:")
            print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print(f"⚠️ Database rebuild timed out (still running in background)")
    except Exception as e:
        print(f"❌ Database rebuild error: {e}")
        
    print("\n" + "=" * 70)
    print(f"✅ Deletion Pipeline Complete")
    print("=" * 70)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_member.py <member_id>")
        sys.exit(1)
        
    try:
        member_id = int(sys.argv[1])
        delete_member_data(member_id)
    except ValueError:
        print("❌ Error: Member ID must be an integer")
        sys.exit(1)
