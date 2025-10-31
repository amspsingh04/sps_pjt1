import nibabel as nib
import sys
import os

# Get the path from the command line
if len(sys.argv) < 2:
    print("Usage: python verify_input.py /path/to/your/ct.nii.gz")
    sys.exit(1)

ct_image_path = sys.argv[1]

if not os.path.exists(ct_image_path):
    print(f"❌ ERROR: File not found at {ct_image_path}")
    sys.exit(1)

try:
    print(f"Checking CT image: {ct_image_path}")
    ct_img = nib.load(ct_image_path)
    ct_shape = ct_img.shape

    print("\n--- RESULTS ---")
    print(f"CT Shape:    {ct_shape}")
    print(f"Dimensions:  {ct_img.ndim}D")
    
    if ct_img.ndim != 3:
        print("\n🔥 This is the problem. Your input CT scan is not a 3D volume.")
    else:
        print("\n✅ This file is 3D. The problem is elsewhere (e.g., in preprocessing.py).")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
