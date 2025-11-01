# postprocess.py
import argparse
import pickle
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os
import sys
from scipy import ndimage
from skimage.morphology import remove_small_objects, label

print("Running Post-processing...")

# --- Post-Processing ("Pink Box") Helpers ---
def post_process_segmentation(mask, min_size=100, smooth_iterations=3):
    """
    Cleans up a binary segmentation mask using the "pink box" logic.
    """
    print(f"   -> Cleaning mask (min size: {min_size}, smoothing: {smooth_iterations} iter)")
    
    labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
    if num_features == 0:
        print("   -> Mask is empty. Nothing to clean.")
        return mask 
        
    cleaned_mask = remove_small_objects(labeled_mask, min_size=min_size)
    
    if np.sum(cleaned_mask) > 0:
        labeled_cleaned_mask = label(cleaned_mask > 0)
        component_sizes = np.bincount(labeled_cleaned_mask.ravel())
        if len(component_sizes) > 1: # Check if any components are left
            largest_component_label = component_sizes[1:].argmax() + 1 
            final_mask = (labeled_cleaned_mask == largest_component_label).astype(np.int16)
        else:
            final_mask = np.zeros_like(cleaned_mask, dtype=np.int16)
    else:
        print("   -> No objects left after size threshold.")
        final_mask = cleaned_mask.astype(np.int16)

    if smooth_iterations > 0 and np.sum(final_mask) > 0:
        print("   -> Smoothing boundaries with morphological closing...")
        final_mask = ndimage.binary_closing(final_mask, iterations=smooth_iterations).astype(np.int16)
    
    return final_mask
# ----------------------------------------------


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Reproject GNN predictions to voxel grid.")
parser.add_argument('--supervoxels_path', type=str, required=True, help="Path to preprocessed_supervoxels.pkl")
parser.add_argument('--predictions_path', type=str, required=True, help="Path to node_predictions.pt")
parser.add_argument('--node_map_path', type=str, required=True, help="Path to node_mapping.pkl")
parser.add_argument('--original_image_nii', type=str, required=True, help="Path to the original NIfTI image (for header info).")
parser.add_argument('--output_nii', type=str, required=True, help="Path to save the final segmentation .nii.gz file.")
parser.add_argument("--min_object_size", type=int, default=100, help="Minimum size (voxels) for an object to be kept")

args = parser.parse_args()

# --- 1. Load All Necessary Files ---
print("   -> Loading files...")
try:
    supervoxel_array = joblib.load(args.supervoxels_path) # Low-res
    node_predictions = torch.load(args.predictions_path).numpy() # Shape [num_nodes, num_classes]

    with open(args.node_map_path, 'rb') as f:
        node_map = pickle.load(f) # Maps {original_sv_id: model_idx}

    # Load original image to get affine, header, AND SHAPE
    original_nii = nib.load(args.original_image_nii)
    original_affine = original_nii.affine
    original_header = original_nii.header
    original_shape = original_nii.shape # e.g., (512, 333, 200)
except FileNotFoundError as e:
    print(f"   -> ❌ FATAL: A required file was not found.", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(1)

# --- 2. Create Inverse Mapping ---
inv_node_map = {v: k for k, v in node_map.items()}
num_nodes, num_classes_pred = node_predictions.shape

# --- 3. Reconstruct Coarse Probability Volume (in LOW-RES space) ---
print("   -> Reconstructing coarse probability volume...")
# This volume is LOW-RES, e.g., (410, 208, 125, C)
prob_volume_lowres = np.zeros(supervoxel_array.shape + (num_classes_pred,), dtype=np.float32)

for model_idx in range(num_nodes):
    if model_idx in inv_node_map:
        original_sv_id = inv_node_map[model_idx]
        probs = node_predictions[model_idx]
        voxel_coords = (supervoxel_array == original_sv_id)
        prob_volume_lowres[voxel_coords] = probs

print("   -> Probability volume created.")

# --- Save the GNN probability map (this is still correct) ---
prob_map_path = args.output_nii.replace(".nii.gz", "_prob_map.nii.gz")
prob_map_nii = nib.Nifti1Image(prob_volume_lowres, original_affine) # Affine is wrong, but shape is what matters
nib.save(prob_map_nii, prob_map_path)
print(f"   -> GNN probability map saved to {prob_map_path}")

# --- 4. Reconstruct Full Segmentation Mask (in LOW-RES space) ---
print("   -> Creating final low-res segmentation mask...")
seg_mask_lowres = np.argmax(prob_volume_lowres, axis=-1).astype(np.int16)

# --- 5. Run "Pink Box" Cleanup (in LOW-RES space) ---
print("--- Running Post-Processing (Pink Box) ---")
final_clean_mask_lowres = np.zeros_like(seg_mask_lowres)
NUM_CLASSES = num_classes_pred

for class_id in range(1, NUM_CLASSES): # Skip background
    class_mask = (seg_mask_lowres == class_id)
    if np.sum(class_mask) == 0: continue
        
    print(f"   -> Cleaning class {class_id}...")
    cleaned_class_mask = post_process_segmentation(
        class_mask, 
        min_size=args.min_object_size, 
        smooth_iterations=2
    )
    final_clean_mask_lowres[cleaned_class_mask > 0] = class_id

print("   -> Post-processing complete.")

# --- 6. ❗ RESAMPLE TO HIGH-RES ❗ ---
print(f"Resampling final mask from {final_clean_mask_lowres.shape} back to original shape {original_shape}...")
# Convert to tensor [B, C, D, H, W] for interpolate
final_mask_tensor = torch.from_numpy(final_clean_mask_lowres).float().unsqueeze(0).unsqueeze(0)
final_mask_resampled = F.interpolate(final_mask_tensor, size=original_shape, mode='nearest')
final_mask_highres = final_mask_resampled.squeeze().numpy().astype(np.int16)
print(f"   -> New shape: {final_mask_highres.shape}")

# --- 7. Save the Final HIGH-RES .nii.gz File ---
print(f"   -> Saving final segmentation to {args.output_nii}")

# --- ❗ THIS IS THE FIX ---
# The variable is 'final_seg_nii', not 'final_nii'
final_seg_nii = nib.Nifti1Image(final_mask_highres, original_affine, original_header)
nib.save(final_seg_nii, args.output_nii)
# --- END FIX ---

print("✅ Post-processing complete.")
