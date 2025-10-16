# generate_labels.py
import argparse
import pickle
import joblib
import numpy as np
import nibabel as nib
from scipy import stats
import networkx as nx

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate labels for supervoxels.")
parser.add_argument('--supervoxels_path', type=str, required=True)
parser.add_argument('--labels_nii_path', type=str, required=True)
parser.add_argument('--graph_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

print("Loading data...")
# --- Load Data Files ---
with open(args.supervoxels_path, 'rb') as f:
    supervoxel_array = joblib.load(f)

label_img = nib.load(args.labels_nii_path)
label_array = label_img.get_fdata()

with open(args.graph_path, 'rb') as f:
    G = pickle.load(f)

# --- ❗ FIX: Define supervoxel_ids from the loaded graph ---
# This line was missing. It's needed to get the list of nodes to process.
supervoxel_ids = list(G.nodes())

print(f"Found {len(supervoxel_ids)} supervoxels.")
print("Aggregating labels...")

supervoxel_labels = {}
for sv_id in supervoxel_ids:
    # Find the coordinates of all voxels belonging to this supervoxel
    voxel_coords = np.where(supervoxel_array == sv_id)
    
    # Use these coordinates to get the labels from the ground-truth file
    if voxel_coords[0].size > 0:
        labels_in_supervoxel = label_array[voxel_coords]
        
        # Find the most frequent label (the mode) and handle ties.
        if labels_in_supervoxel.size > 0:
            most_common_label = int(stats.mode(labels_in_supervoxel, keepdims=False)[0][0])
            supervoxel_labels[sv_id] = most_common_label
        else:
            supervoxel_labels[sv_id] = 0 # Default if no labels found
    else:
        # Handle cases where a supervoxel ID from the graph might not be in the array
        supervoxel_labels[sv_id] = 0 # Default to background

# --- Save the Output ---
with open(args.output_path, 'wb') as f:
    pickle.dump(supervoxel_labels, f)

print(f"✅ Label mapping saved successfully to {args.output_path}")
