import argparse
import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic
import pandas as pd
import joblib

def load_nifti(path):
    nifti_img = nib.load(path)
    volume = nifti_img.get_fdata()
    spacing = nifti_img.header.get_zooms()
    return volume, spacing

def resample_volume(volume, original_spacing, new_spacing=(1.0, 1.0, 1.0)):
    sitk_img = sitk.GetImageFromArray(volume)
    sitk_img.SetSpacing(tuple(float(s) for s in original_spacing))

    original_size = sitk_img.GetSize()
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
    if std == 0:
        std = 1e-6  # avoid division by zero
    volume = (volume - mean) / std
    return volume

def compute_supervoxels(volume, n_segments=500, compactness=0.1):
    if volume.ndim != 3:
        raise ValueError(f"Input volume must be 3D, got shape {volume.shape}")
    norm_volume = (volume - np.min(volume)) / (np.ptp(volume))  # ptp = max-min
    labels = slic(norm_volume, n_segments=n_segments, compactness=compactness, start_label=1)
    assert labels.shape == volume.shape, f"Supervoxels shape {labels.shape} != volume shape {volume.shape}"

    return labels
def compute_supervoxel_features(volume, labels):
    regions = np.unique(labels)
    features = []

    for label in regions:
        mask = (labels == label)
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

    args = parser.parse_args()

    # Make sure output directory exists
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Loading volume...")
    volume, spacing = load_nifti(args.input_path)
    print(f"Original volume shape: {volume.shape}, spacing: {spacing}")

    print("Resampling volume...")
    resampled = resample_volume(volume, spacing)
    print(f"Resampled volume shape: {resampled.shape}")

    print("Preprocessing (clipping & normalization)...")
    preprocessed = preprocess(resampled)

    print("Computing supervoxels...")
    supervoxels = compute_supervoxels(preprocessed)

    print(f"Supervoxels shape: {supervoxels.shape}")
    assert supervoxels.shape == preprocessed.shape, "Supervoxels shape must match volume shape"

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
