
import shutil
from pathlib import Path

RAW_ROOT = Path(r"C:\Users\sanch\Downloads\MURA-v1.1\train\XR_HUMERUS")
OUT_ROOT = Path("dataset_sorted")

NORMAL_DIR = OUT_ROOT / "normal"
OSTEO_DIR = OUT_ROOT / "osteoporotic"

NORMAL_DIR.mkdir(parents=True, exist_ok=True)
OSTEO_DIR.mkdir(parents=True, exist_ok=True)

# Check if the source directory exists
if not RAW_ROOT.exists():
    print(f"❌ Error: Source directory does not exist: {RAW_ROOT}")
    print("Please check the path and make sure MURA dataset is downloaded and extracted.")
    exit(1)

print(f"📂 Processing directory: {RAW_ROOT}")
patient_count = 0
normal_count = 0
osteo_count = 0
skipped_count = 0

for patient_dir in RAW_ROOT.iterdir():
    if not patient_dir.is_dir():
        print(f"⚠️  Skipping non-directory: {patient_dir.name}")
        continue
    
    patient_count += 1
    print(f"👤 Processing patient {patient_count}: {patient_dir.name}")

    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            print(f"   ⚠️  Skipping non-directory in patient: {study_dir.name}")
            continue

        print(f"   📁 Study directory: {study_dir.name}")
        
        if "negative" in study_dir.name.lower():
            target_dir = NORMAL_DIR
            category = "normal"
        elif "positive" in study_dir.name.lower():
            target_dir = OSTEO_DIR
            category = "osteoporotic"
        else:
            print(f"   ⚠️  Skipping study (no positive/negative): {study_dir.name}")
            skipped_count += 1
            continue

        images_in_study = list(study_dir.glob("*.png"))
        print(f"   🖼️  Found {len(images_in_study)} images in {category} study")
        
        for img_file in images_in_study:
            # Create unique filename using patient and study info
            unique_name = f"{patient_dir.name}_{study_dir.name}_{img_file.name}"
            target_path = target_dir / unique_name
            
            try:
                shutil.copy2(img_file, target_path)
                if category == "normal":
                    normal_count += 1
                else:
                    osteo_count += 1
            except Exception as e:
                print(f"   ❌ Error copying {img_file.name}: {e}")

print(f"\n📊 Summary:")
print(f"   Patients processed: {patient_count}")
print(f"   Normal images: {normal_count}")
print(f"   Osteoporotic images: {osteo_count}")
print(f"   Studies skipped: {skipped_count}")
print("✅ Sorting complete.")
