# postprocess.py
import argparse
import pickle
import joblib
import torch
import numpy as np
import nibabel as nib
import os

print("Running Post-processing...")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Reproject GNN predictions to voxel grid.")
parser.add_argument('--supervoxels_path', type=str, required=True, help="Path to preprocessed_supervoxels.pkl")
parser.add_argument('--predictions_path', type=str, required=True, help="Path to node_predictions.pt")
parser.add_argument('--node_map_path', type=str, required=True, help="Path to node_mapping.pkl")
parser.add_argument('--original_image_nii', type=str, required=True, help="Path to the original NIfTI image (for header info).")
parser.add_argument('--output_nii', type=str, required=True, help="Path to save the final segmentation .nii.gz file.")
args = parser.parse_args()

# --- 1. Load All Necessary Files ---
print("   -> Loading files...")
supervoxel_array = joblib.load(args.supervoxels_path)
node_predictions = torch.load(args.predictions_path).numpy() # Shape [num_nodes, num_classes]

with open(args.node_map_path, 'rb') as f:
    node_map = pickle.load(f) # Maps {original_sv_id: model_idx}

# Load original image to get affine and header
original_nii = nib.load(args.original_image_nii)
affine = original_nii.affine
header = original_nii.header

# --- 2. Create Inverse Mapping ---
# We need to map {model_idx: original_sv_id}
inv_node_map = {v: k for k, v in node_map.items()}
num_nodes, num_classes = node_predictions.shape

# --- 3. Reconstruct Coarse Probability Volume ---
# This is steps 2, 3, and 4 from your list.
# We create a new 4D volume with shape [D, H, W, num_classes]
print("   -> Reconstructing coarse probability volume...")
prob_volume = np.zeros(supervoxel_array.shape + (num_classes,), dtype=np.float32)

for model_idx in range(num_nodes):
    original_sv_id = inv_node_map[model_idx]
    
    # Get the probability vector for this node
    probs = node_predictions[model_idx] # Shape [num_classes]
    
    # Find all voxels belonging to this supervoxel
    voxel_coords = (supervoxel_array == original_sv_id)
    
    # Assign the node's probabilities to all its voxels
    prob_volume[voxel_coords] = probs

print("   -> Probability volume created.")

prob_map_path=args.output_nii.replace(".nii.gz","_prob_map.nii.gz")
prob_map_nii=nib.Nifti1Image(prob_volume,affine,header)
nib.save(prob_map_nii,prob_map_path)
print(f"   -> GNN probability map saved to {prob_map_path}")

# --- 4. Reconstruct Full Segmentation Mask ---
# This is step 5. We take the argmax over the class dimension.
print("   -> Creating final segmentation mask...")
seg_mask = np.argmax(prob_volume, axis=-1).astype(np.int16)

# --- 5. Save the Final .nii.gz File ---
print(f"   -> Saving final segmentation to {args.output_nii}")
# Create a new NIfTI image using the mask and the original's spatial info
final_seg_nii = nib.Nifti1Image(seg_mask, affine, header)
nib.save(final_seg_nii, args.output_nii)

print("✅ Post-processing complete.")
