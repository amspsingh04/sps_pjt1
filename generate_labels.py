import pickle
import numpy as np
import nibabel as nib 
from scipy import stats
import networkx as nx
import joblib

print("Loading data...")
with open('output/preprocessed_supervoxels.pkl', 'rb') as f:
    supervoxel_array = joblib.load(f)

label_img = nib.load('dataset/labelsTr/hippocampus_001.nii.gz')
label_array = label_img.get_fdata()

with open('output/supervoxel_graph.gpickle', 'rb') as f:
    G = pickle.load(f)
supervoxel_ids = list(G.nodes())

print(f"Found {len(supervoxel_ids)} supervoxels.")
print("Aggregating labels...")

supervoxel_labels = {}
for sv_id in supervoxel_ids:
    voxel_coords = np.where(supervoxel_array == sv_id)
    
    labels_in_supervoxel = label_array[voxel_coords]
    
    
    if labels_in_supervoxel.size > 0:
        most_common_label = int(stats.mode(labels_in_supervoxel, keepdims=False)[0][0]) 
        supervoxel_labels[sv_id] = most_common_label
    else:
        supervoxel_labels[sv_id] = 0 
        
output_path = 'output/supervoxel_label_mapping.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(supervoxel_labels, f)

print(f"Label mapping saved successfully to {output_path}")