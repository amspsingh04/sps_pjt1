"""
napari_viewing.py — Simple NIfTI viewer with napari
pip install napari[all] nibabel PyQt5
"""
import napari
import nibabel as nib
import numpy as np
import os
import sys

print("🔄 Launching napari... please wait (a few seconds)")

def load_nifti(path):
    print(f"📂 Loading: {path}")
    img = nib.load(path)
    data = img.get_fdata()
    print(f"✅ Loaded shape: {data.shape}, dtype: {data.dtype}")
    return data

ct_path = r" "
seg_path = r" "

if not (os.path.exists(ct_path)):
    print(f"❌ Could not find {ct_path}.nii.gz in current directory.")
    sys.exit(1)
elif not (os.path.exists(seg_path)):
    print(f"❌ Could not find {ct_path}.nii.gz in current directory.")
    sys.exit(1)

ct_data = load_nifti(ct_path)
seg_data = load_nifti(seg_path)

viewer = napari.Viewer()

viewer.add_image(
    ct_data,
    name="CT",
    colormap="gray",
    blending="additive",
    contrast_limits=[np.min(ct_data), np.max(ct_data)],
)


viewer.add_labels(
    seg_data.astype(np.int32),
    name="Labels",
    opacity=0.5,
)

print("🚀 Opening napari window...")
napari.run() 
