import nibabel as nib
import sys

# --- 1. SET YOUR FILE PATHS HERE ---
# (I'm guessing the paths based on your last error)

ct_image_path = "PanTS/data/ImageTr/PanTS_00000002/ct.nii.gz"
label_image_path = "PanTS/data/labels/PanTS_00000002/combined_labels.nii.gz" 
# ^^^ Note: I put 'labelsTr' here. If that's wrong, change it back to 'labels'.

# -----------------------------------

try:
    # Load the images
    print(f"Checking CT image: {ct_image_path}")
    ct_img = nib.load(ct_image_path)
    
    print(f"Checking Label image: {label_image_path}")
    label_img = nib.load(label_image_path)

    # Get the shapes
    ct_shape = ct_img.shape
    label_shape = label_img.shape

    print("\n--- RESULTS ---")
    print(f"CT Shape:    {ct_shape}")
    print(f"Label Shape: {label_shape}")
    
    # Compare them
    if ct_shape == label_shape:
        print("\n✅ SUCCESS: The shapes match perfectly.")
    else:
        print("\n❌ ERROR: The shapes DO NOT match.")
        print("This is the source of your error.")

except FileNotFoundError as e:
    print(f"\n❌ FAILED: Could not find a file.")
    print(e)
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
