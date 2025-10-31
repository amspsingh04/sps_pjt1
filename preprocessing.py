# preprocessing.py
import argparse
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic
import pandas as pd
import joblib
import os
import sys

def load_nifti(path):
    nifti_img = nib.load(path)
    volume = nifti_img.get_fdata()
    
    # --- ❗ THIS IS THE FIX ---
    # The header is lying about dimensions. get_zooms() might return a 2-tuple.
    # We know the volume is 3D, so we MUST have 3 spacing values.
    spacing = nifti_img.header.get_zooms()
    full_spacing = list(spacing[:3])  # Get up to 3 values (x, y, z)
    
    # If the header only gave 1 or 2, fill the rest with 1.0
    while len(full_spacing) < 3:
        full_spacing.append(1.0)
    
    return volume, tuple(full_spacing)  # Guarantees a 3-tuple
    # --- END FIX ---

def resample_volume(volume, original_spacing, new_spacing=(1.0, 1.0, 1.0)):
    sitk_img = sitk.GetImageFromArray(volume)
    
    if len(original_spacing) != 3:
        print(f"   -> ❌ FATAL: Resample received non-3D spacing: {original_spacing}", file=sys.stderr)
        raise ValueError("Resampling requires 3D spacing.")
        
    sitk_img.SetSpacing(tuple(float(s) for s in original_spacing))
    original_size = sitk_img.GetSize()

    if len(original_size) != 3:
        print(f"   -> ❌ FATAL: SITK image is not 3D (size: {original_size}). Numpy shape was: {volume.shape}", file=sys.stderr)
        raise ValueError("Resampling requires 3D volume.")

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputDirection(sitk_img.GetDirection())

    resampled = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled)


def preprocess(volume, hu_min=-1000, hu_max=400):
    volume = np.clip(volume, hu_min, hu_max)
    mean = np.mean(volume)
    std = np.std(volume)
    if std > 0:
        volume = (volume - mean) / std
    return volume


def compute_supervoxels(volume, n_segments=500, compactness=0.1):
    # Normalize volume for SLIC
    norm_volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)
    
    # slic function is 3D-aware. If volume is 3D, output is 3D.
    labels = slic(norm_volume, n_segments=n_segments, compactness=compactness, start_label=1, enforce_connectivity=True, channel_axis=None)
    return labels


def compute_supervoxel_features(volume, labels):
    regions = np.unique(labels)
    features = []

    for label in regions:
        if label == 0: continue # slic starts at 1
        mask = (labels == label)
        if np.sum(mask) == 0: continue # Skip empty masks
        
        mean_intensity = np.mean(volume[mask])
        std_intensity = np.std(volume[mask])
        voxel_count = np.sum(mask)

        features.append({
            "supervoxel_id": int(label),
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "voxel_count": int(voxel_count),
        })

    return pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3D CT volume & compute supervoxels")
    parser.add_argument("input_path", type=str, help="Path to .nii.gz file")
    parser.add_argument("--out_prefix", type=str, default="output/preprocessed", help="Prefix for saved outputs")
    parser.add_argument("--n_segments", type=int, default=500, help="Number of supervoxels")

    args = parser.parse_args()
    out_dir=os.path.dirname(args.out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("Loading volume...")
    volume, spacing = load_nifti(args.input_path)
    print(f"   -> Original shape: {volume.shape}, spacing: {spacing}")

    if volume.ndim != 3:
        print(f"   -> ❌ FATAL: Input volume is not 3D. Shape is {volume.shape}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Resampling volume...")
    resampled = resample_volume(volume, spacing)
    print(f"   -> Resampled shape: {resampled.shape}")

    print("Preprocessing (clipping & normalization)...")
    preprocessed = preprocess(resampled)

    print("Computing supervoxels...")
    supervoxels = compute_supervoxels(preprocessed, n_segments=args.n_segments)
    print(f"   -> Supervoxel array shape: {supervoxels.shape}")

    print("Computing supervoxel features...")
    features_df = compute_supervoxel_features(preprocessed, supervoxels)

    print("Done. Sample features:")
    print(features_df.head())

    print(f"Total number of supervoxels: {features_df['supervoxel_id'].nunique()}")

    # Save outputs
    joblib.dump(preprocessed, f"{args.out_prefix}_volume.pkl")
    joblib.dump(supervoxels, f"{args.out_prefix}_supervoxels.pkl")
    features_df.to_csv(f"{args.out_prefix}_features.csv", index=False)

    print(f"Saved preprocessed volume, supervoxels, and features with prefix '{args.out_prefix}'")


if __name__ == "__main__":
    main()
