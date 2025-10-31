# generate_labels.py
import argparse
import pickle
import joblib
import numpy as np
import nibabel as nib
from scipy import stats
import networkx as nx
import sys 

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate labels for supervoxels.")
parser.add_argument('--supervoxels_path', type=str, required=True)
parser.add_argument('--labels_nii_path', type=str, required=True)
parser.add_argument('--graph_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

print("Loading data...")
# --- Load Data Files ---
try:
    with open(args.supervoxels_path, 'rb') as f:
        supervoxel_array = joblib.load(f)
except FileNotFoundError:
    print(f"   -> ❌ FATAL: Cannot find supervoxel file: {args.supervoxels_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"   -> ❌ FATAL: Error loading supervoxel file: {e}", file=sys.stderr)
    sys.exit(1)

try:
    label_img = nib.load(args.labels_nii_path)
    label_array = label_img.get_fdata()
except FileNotFoundError:
    print(f"   -> ❌ FATAL: Cannot find label file: {args.labels_nii_path}", file=sys.stderr)
    sys.exit(1)

label_shape = label_array.shape
sv_shape = supervoxel_array.shape

# --- Robust Shape Checking ---
if label_array.ndim != 3 or supervoxel_array.ndim != 3:
    print(f"   -> ❌ FATAL: Mismatched dimensions. Both arrays must be 3D.", file=sys.stderr)
    print(f"   -> Label shape: {label_shape} (Dimensions: {label_array.ndim}D)", file=sys.stderr)
    print(f"   -> Supervoxel shape: {sv_shape} (Dimensions: {supervoxel_array.ndim}D)", file=sys.stderr)
    print("   -> This error originates in 'preprocessing.py'.", file=sys.stderr)
    sys.exit(1) # Exit with an error

# --- Clipping Logic (now that we know both are 3D) ---
D_min = min(label_shape[0], sv_shape[0])
H_min = min(label_shape[1], sv_shape[1])
W_min = min(label_shape[2], sv_shape[2])

if label_shape != sv_shape:
    print(f"   -> Warning: Mismatched 3D shapes detected (due to padding/resampling).")
    print(f"   -> Label shape: {label_shape}, Supervoxel shape: {sv_shape}")
    print(f"   -> Clipping both to common shape ({D_min}, {H_min}, {W_min})")
    supervoxel_array = supervoxel_array[:D_min, :H_min, :W_min]
    label_array = label_array[:D_min, :H_min, :W_min]
# --- End of Shape Fix ---

try:
    with open(args.graph_path, 'rb') as f:
        G = pickle.load(f)
except FileNotFoundError:
     print(f"   -> ❌ FATAL: Cannot find graph file: {args.graph_path}", file=sys.stderr)
     sys.exit(1)

supervoxel_ids = list(G.nodes())

print(f"Found {len(supervoxel_ids)} supervoxels.")
print("Aggregating labels...")

supervoxel_labels = {}
for sv_id in supervoxel_ids:
    voxel_coords = np.where(supervoxel_array == sv_id)
    
    if voxel_coords[0].size > 0:
        labels_in_supervoxel = label_array[voxel_coords]
        
        if labels_in_supervoxel.size > 0:
            # --- ❗ THIS IS THE FIX ---
            # 1. Get the mode result (which could be a scalar OR an array)
            mode_result = stats.mode(labels_in_supervoxel, keepdims=False)[0]
            
            # 2. Force it to be an array, flatten it, and take the first element
            # This handles both scalars and arrays robustly.
            most_common_label = int(np.asarray(mode_result).flatten()[0])
            # --- END FIX ---
            
            supervoxel_labels[sv_id] = most_common_label
        else:
            supervoxel_labels[sv_id] = 0 # Default if no labels found
    else:
        supervoxel_labels[sv_id] = 0 # Default to background

# --- Save the Output ---
with open(args.output_path, 'wb') as f:
    pickle.dump(supervoxel_labels, f)

print(f"✅ Label mapping saved successfully to {args.output_path}")
