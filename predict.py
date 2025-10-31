# predict.py
"""
The Grand Unified Inference Script.
This script chains the entire GNN + CNN + Post-processing pipeline
to predict a segmentation mask from a single, new CT scan.
"""

import argparse
import os
import sys
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
from scipy import stats, ndimage

import torch
import torch.nn as nn  # <--- Imported as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TopKPooling

import timm
from torchvision import transforms
import SimpleITK as sitk
from skimage.segmentation import slic
from skimage.morphology import remove_small_objects, label

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
REFINER_PATCH_SIZE = (96, 96, 96)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================================================
# SECTION 1: MODEL CLASS DEFINITIONS
# =====================================================================================

# --- ❗ FIX 1: Inherit from nn.Module ---
class GNNUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, pool_ratio=0.5, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.pools.append(TopKPooling(hidden_channels * heads, ratio=pool_ratio))
        for _ in range(num_layers - 1):
            self.down_convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            self.pools.append(TopKPooling(hidden_channels * heads, ratio=pool_ratio))
        self.up_convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.up_convs.append(GATConv(hidden_channels * heads * 2, hidden_channels, heads=heads))
        self.up_convs.append(GATConv(hidden_channels * heads * 2, hidden_channels, heads=heads))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    def forward(self, x, edge_index, batch=None):
        if batch is None: batch = edge_index.new_zeros(x.size(0))
        skip_connections_x = []
        skip_connections_edge_index = []
        skip_connections_perm = []
        for i in range(self.num_layers):
            x = F.relu(self.down_convs[i](x, edge_index))
            skip_connections_x.append(x); skip_connections_edge_index.append(edge_index)
            x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, None, batch)
            skip_connections_perm.append(perm)
        skip_connections_x.reverse(); skip_connections_edge_index.reverse(); skip_connections_perm.reverse()
        for i in range(self.num_layers):
            x_skip = skip_connections_x[i]; edge_index = skip_connections_edge_index[i]; perm = skip_connections_perm[i]
            unpooled_x = x.new_zeros(x_skip.size(0), x.size(1)); unpooled_x[perm] = x; x = unpooled_x
            x = torch.cat([x, x_skip], dim=1)
            x = F.relu(self.up_convs[i](x, edge_index))
        logits = self.output_layer(x)
        return F.log_softmax(logits, dim=-1)

# --- ❗ FIX 1: Inherit from nn.Module ---
class Simple3DRefiner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.conv_stack(x)


# =====================================================================================
# SECTION 2: HELPER FUNCTIONS
# (No changes in this section)
# =====================================================================================

# --- Preprocessing Helpers ---
def load_nifti_for_predict(path):
    nifti_img = nib.load(path)
    volume = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header
    spacing = nifti_img.header.get_zooms()
    full_spacing = list(spacing[:3])
    while len(full_spacing) < 3:
        full_spacing.append(1.0)
    return volume, tuple(full_spacing), affine, header

def resample_volume(volume, original_spacing, new_spacing=(1.0, 1.0, 1.0)):
    sitk_img = sitk.GetImageFromArray(volume)
    if len(original_spacing) != 3: raise ValueError("Resampling requires 3D spacing.")
    sitk_img.SetSpacing(tuple(float(s) for s in original_spacing))
    original_size = sitk_img.GetSize()
    if len(original_size) != 3: raise ValueError("Resampling requires 3D volume.")
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing); resample.SetSize(new_size); resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputOrigin(sitk_img.GetOrigin()); resample.SetOutputDirection(sitk_img.GetDirection())
    resampled = resample.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled)

def preprocess(volume, hu_min=-1000, hu_max=400):
    volume = np.clip(volume, hu_min, hu_max)
    mean = np.mean(volume); std = np.std(volume)
    if std > 0: volume = (volume - mean) / std
    return volume

def compute_supervoxels(volume, n_segments=500, compactness=0.1):
    norm_volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)
    labels = slic(norm_volume, n_segments=n_segments, compactness=compactness, start_label=1, enforce_connectivity=True, channel_axis=None)
    return labels

# --- Graph Building Helpers ---
def _extract_dino_feature_2d(patch_2d, model, transform):
    img_3ch = np.stack([patch_2d]*3, axis=-1).astype(np.uint8)
    input_tensor = transform(img_3ch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.forward_features(input_tensor)
    cls_token_feature = feat[:, 0]
    return cls_token_feature.squeeze().cpu().numpy()

def _get_supervoxel_2d_patch(volume, supervoxels, sv_id):
    slices_idx = np.where(np.any(supervoxels == sv_id, axis=(1, 2)))[0]
    if len(slices_idx) == 0: return None
    slice_idx = slices_idx[len(slices_idx)//2]
    mask_2d = supervoxels[slice_idx] == sv_id
    patch = volume[slice_idx] * mask_2d
    return patch

def _compute_spatial_features(supervoxels):
    spatial_feats = []
    for label in np.unique(supervoxels):
        if label == 0: continue
        coords = np.array(np.where(supervoxels == label))
        centroid = coords.mean(axis=1)
        size = coords.ptp(axis=1) + 1
        spatial_feats.append({
            "supervoxel_id": label, "centroid_z": centroid[0], "centroid_y": centroid[1], "centroid_x": centroid[2],
            "size_z": size[0], "size_y": size[1], "size_x": size[2],
        })
    return pd.DataFrame(spatial_feats)

def _get_adjacency(supervoxels):
    structure = ndimage.generate_binary_structure(3, 1)
    adjacency = {}
    labels = np.unique(supervoxels)
    for label in labels:
        if label == 0: continue
        mask = (supervoxels == label)
        dilated = ndimage.binary_dilation(mask, structure=structure)
        neighbors = np.unique(supervoxels[dilated])
        neighbors = neighbors[neighbors != label]
        neighbors = neighbors[neighbors != 0]
        adjacency[label] = neighbors.tolist()
    return adjacency

# --- Refiner Helpers ---
def pad_or_crop_to_size(data, target_shape):
    current_shape = np.array(data.shape)
    spatial_dims = current_shape[-3:]
    target_spatial_shape = target_shape
    shape_diff = target_spatial_shape - spatial_dims
    crop_pre = np.maximum(0, -shape_diff) // 2
    crop_post = np.maximum(0, -shape_diff) - crop_pre
    if data.ndim == 4: # 4D Input
        slices = (slice(None), slice(crop_pre[0], current_shape[1] - crop_post[0]),
                  slice(crop_pre[1], current_shape[2] - crop_post[1]), slice(crop_pre[2], current_shape[3] - crop_post[2]))
    else: # 3D Mask
        slices = (slice(crop_pre[0], current_shape[0] - crop_post[0]),
                  slice(crop_pre[1], current_shape[1] - crop_post[1]), slice(crop_pre[2], current_shape[2] - crop_post[2]))
    data = data[slices]
    pad_pre = np.maximum(0, shape_diff) // 2
    pad_post = np.maximum(0, shape_diff) - pad_pre
    if data.ndim == 4: # 4D Input
        pad_width = ((0, 0), (pad_pre[0], pad_post[0]), (pad_pre[1], pad_post[1]), (pad_pre[2], pad_post[2]))
    else: # 3D Mask
        pad_width = ((pad_pre[0], pad_post[0]), (pad_pre[1], pad_post[1]), (pad_pre[2], pad_post[2]))
    data = np.pad(data, pad_width, mode='constant', constant_values=0)
    return data

# --- Post-Processing ("Pink Box") Helpers ---
def post_process_segmentation(mask, min_size=100, smooth_iterations=3):
    """
    Cleans up a binary segmentation mask using the "pink box" logic.
    """
    print(f"   -> Cleaning mask (min size: {min_size}, smoothing: {smooth_iterations} iter)")
    
    # 1. Keep Largest Connected Component & Remove Tiny Objects
    labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
    
    if num_features == 0:
        print("   -> Mask is empty. Nothing to clean.")
        return mask # Return empty mask
        
    cleaned_mask = remove_small_objects(labeled_mask, min_size=min_size)
    
    if np.sum(cleaned_mask) > 0:
        labeled_cleaned_mask = label(cleaned_mask > 0)
        component_sizes = np.bincount(labeled_cleaned_mask.ravel())
        largest_component_label = component_sizes[1:].argmax() + 1 # (skip background label 0)
        final_mask = (labeled_cleaned_mask == largest_component_label).astype(np.int16)
    else:
        print("   -> No objects left after size threshold.")
        final_mask = cleaned_mask.astype(np.int16)

    # 2. Smooth Boundaries (Morphological Closing)
    if smooth_iterations > 0:
        print("   -> Smoothing boundaries with morphological closing...")
        final_mask = ndimage.binary_closing(final_mask, iterations=smooth_iterations).astype(np.int16)
    
    return final_mask


# =====================================================================================
# SECTION 3: MAIN INFERENCE FUNCTION
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Full GNN+CNN Inference Pipeline")
    parser.add_argument("--input_ct", type=str, required=True, help="Path to the new, unseen CT.nii.gz file.")
    parser.add_argument("--output_nii", type=str, required=True, help="Path to save the final segmentation.nii.gz file.")
    parser.add_argument("--gnn_model_path", type=str, required=True, help="Path to the trained GNN (e.g., output/run_.../best_model.pt)")
    parser.add_argument("--refiner_model_path", type=str, required=True, help="Path to the trained refiner (my_best_refiner.pt)")
    parser.add_argument("--n_segments", type=int, default=400, help="Number of supervoxels to generate (must match training)")
    parser.add_argument("--min_object_size", type=int, default=100, help="Minimum size (voxels) for an object to be kept")
    # Add arguments for our "hardcoded guesses"
    parser.add_argument("--num_gnn_classes", type=int, default=5, help="Number of classes the GNN model was trained to output.")
    parser.add_argument("--num_refiner_classes", type=int, default=5, help="Number of classes the Refiner model was trained to output.")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    
    # --- PHASE 1A: PREPROCESSING (from preprocessing.py) ---
    print("--- PHASE 1A: Preprocessing ---")
    print(f"Loading {args.input_ct}...")
    volume, spacing, original_affine, original_header = load_nifti_for_predict(args.input_ct)
    print(f"   -> Original shape: {volume.shape}")

    print("Resampling...")
    resampled_vol = resample_volume(volume, spacing)
    print(f"   -> Resampled shape: {resampled_vol.shape}")
    
    print("Normalizing...")
    preprocessed_vol = preprocess(resampled_vol)
    
    print("Computing supervoxels...")
    supervoxels_vol = compute_supervoxels(preprocessed_vol, n_segments=args.n_segments)
    print(f"   -> Supervoxel array shape: {supervoxels_vol.shape}")

    # --- PHASE 1B: GRAPH BUILDING (from graph_build.py) ---
    print("--- PHASE 1B: Graph Building ---")
    print("Loading DINO model...")
    dino_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(DEVICE)
    dino_model.eval()
    dino_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
    ])

    print("Extracting DINO features (this may take a while)...")
    dino_features_list = []
    all_sv_ids = np.unique(supervoxels_vol)
    all_sv_ids = all_sv_ids[all_sv_ids != 0] # Remove background
    
    for sv_id in all_sv_ids:
        patch = _get_supervoxel_2d_patch(preprocessed_vol, supervoxels_vol, sv_id)
        if patch is None or patch.max() == 0:
            dino_feat = np.zeros(768)
        else:
            dino_feat = _extract_dino_feature_2d(patch, dino_model, dino_transform)
        dino_features_list.append(dino_feat)
    
    dino_features_array = np.vstack(dino_features_list)
    dino_df = pd.DataFrame(dino_features_array, index=all_sv_ids).add_prefix('dino_feat_')
    
    print("Extracting spatial features...")
    spatial_df = _compute_spatial_features(supervoxels_vol)
    
    print("Combining features...")
    spatial_df = spatial_df.set_index('supervoxel_id')
    features_df = spatial_df.join(dino_df).fillna(0) # Join on index
    
    print("Computing adjacency...")
    adjacency = _get_adjacency(supervoxels_vol)
    
    print("Building NetworkX graph...")
    G = nx.Graph()
    feat_dict = {sv_id: row.values for sv_id, row in features_df.iterrows()}
    
    for sv_id, features in feat_dict.items():
        if sv_id in features_df.index: # Ensure sv_id is valid
            G.add_node(sv_id, features=features)
        
    for node, neighbors in adjacency.items():
        if node in G.nodes(): # Ensure source node is valid
            for n in neighbors:
                if n in G.nodes(): # Ensure neighbor is a valid node
                    G.add_edge(node, n)
    
    print(f"   -> Graph built with {G.number_of_nodes()} nodes.")
    
    # --- PHASE 1C: GNN INFERENCE (from train_unet.py) ---
    print("--- PHASE 1C: GNN Inference ---")
    
    # 1. Convert NetworkX graph to PyG Data
    node_list = sorted(list(G.nodes()))
    node_map = {node: i for i, node in enumerate(node_list)}
    inv_node_map = {i: node for node, i in node_map.items()}

    node_features = [torch.tensor(G.nodes[n]['features']) for n in node_list]
    x = torch.stack(node_features, dim=0).float().to(DEVICE)
    
    remapped_edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous().to(DEVICE)
    
    data = Data(x=x, edge_index=edge_index)

    # 2. Load GNN Model
    print(f"Loading GNN model from {args.gnn_model_path}...")
    num_node_features = data.num_node_features
    NUM_GNN_CLASSES = args.num_gnn_classes # Use argument
    
    gnn_model = GNNUNet(
        in_channels=num_node_features,
        hidden_channels=64, # Must match the model you saved
        out_channels=NUM_GNN_CLASSES,
        heads=4 # Must match the model you saved
    ).to(DEVICE)
    gnn_model.load_state_dict(torch.load(args.gnn_model_path, map_location=DEVICE))
    gnn_model.eval()
    
    # 3. Run GNN Inference
    print("Running GNN inference...")
    with torch.no_grad():
        log_probs = gnn_model(data.x, data.edge_index)
        probs = torch.exp(log_probs).cpu().numpy() # Shape [num_nodes, num_classes]

    # 4. Reproject to create Coarse Probability Map (from postprocess.py)
    print("Reprojecting GNN predictions to 3D coarse map...")
    coarse_prob_map = np.zeros(supervoxels_vol.shape + (NUM_GNN_CLASSES,), dtype=np.float32)

    for model_idx in range(probs.shape[0]):
        if model_idx in inv_node_map: # Check if key exists
            original_sv_id = inv_node_map[model_idx]
            node_probs = probs[model_idx] # Shape [num_classes]
            voxel_coords = (supervoxels_vol == original_sv_id)
            coarse_prob_map[voxel_coords] = node_probs
    
    print(f"   -> Coarse map shape: {coarse_prob_map.shape}")
    
    # --- PHASE 2: CNN REFINEMENT (from train_refiner.py) ---
    print("--- PHASE 2: CNN Refinement ---")
    
    # 1. Load Refiner Model
    print(f"Loading Refiner model from {args.refiner_model_path}...")
    NUM_REFINER_CLASSES = args.num_refiner_classes # Use argument
    
    # --- ❗ FIX 2: Correct variable assignment ---
    NUM_REFINER_IN_CHANNELS = 1 + NUM_REFINER_CLASSES # 1 for CT
    
    refiner_model = Simple3DRefiner(
        in_channels=NUM_REFINER_IN_CHANNELS,
        out_channels=NUM_REFINER_CLASSES
    ).to(DEVICE)
    refiner_model.load_state_dict(torch.load(args.refiner_model_path, map_location=DEVICE))
    refiner_model.eval()
    
    # 2. Resample original CT to match coarse map
    print("Resampling original CT to match coarse map...")
    # We use 'preprocessed_vol' which is already resampled, normalized, and 3D
    ct_tensor = torch.from_numpy(preprocessed_vol).float().unsqueeze(0).unsqueeze(0)
    ct_resampled = F.interpolate(ct_tensor, size=coarse_prob_map.shape[:-1], mode='trilinear', align_corners=False)
    ct_resampled_np = ct_resampled.squeeze().numpy() # [D, H, W]
    
    # 3. Stack CT and GNN Probs as channels
    gnn_map_ch_first = np.moveaxis(coarse_prob_map, -1, 0) # [C, D, H, W]
    ct_map_ch_first = np.expand_dims(ct_resampled_np, axis=0) # [1, D, H, W]
    
    # --- Check for channel mismatch ---
    if gnn_map_ch_first.shape[0] != NUM_GNN_CLASSES:
         print(f"   -> ❌ FATAL: GNN prob map has {gnn_map_ch_first.shape[0]} channels, but model expects {NUM_GNN_CLASSES}.")
         sys.exit(1)
         
    refiner_input_stack = np.concatenate([ct_map_ch_first, gnn_map_ch_first], axis=0)
    print(f"   -> Refiner input stack shape: {refiner_input_stack.shape}")
    
    # 4. Find Bounding Box to crop
    coarse_mask = np.argmax(coarse_prob_map, axis=-1)
    bbox_slices = ndimage.find_objects(coarse_mask > 0)
    
    if not bbox_slices:
        print("   -> GNN predicted an empty mask. Skipping refinement.")
        refined_mask = coarse_mask
    else:
        print("   -> Found bounding box. Cropping and running refinement...")
        # We need to find the *largest* bounding box
        bbox = bbox_slices[0] # Just take the first one for simplicity
        
        # Crop the input stack
        # Input is [C, D, H, W]
        slices = (slice(None), bbox[0], bbox[1], bbox[2])
        input_patch = refiner_input_stack[slices]
        
        # 5. Pad/Crop patch to match Refiner's training size
        input_patch_padded = pad_or_crop_to_size(input_patch, REFINER_PATCH_SIZE)
        
        # 6. Run Refiner Inference
        input_tensor = torch.from_numpy(input_patch_padded).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            refined_logits_patch = refiner_model(input_tensor) # [1, C, D, H, W]
        
        # 7. "Un-pad" / "Un-crop" the prediction
        refined_logits_patch_np = refined_logits_patch.squeeze().cpu().numpy()
        
        original_patch_shape = input_patch.shape # [C, D, H, W]
        refined_logits_unpadded = pad_or_crop_to_size(refined_logits_patch_np, original_patch_shape[1:])
        
        # Create a full-sized output volume
        refined_logits_volume = np.zeros((NUM_REFINER_CLASSES,) + coarse_prob_map.shape[:-1], dtype=np.float32)
        
        # Place the patch back in its bounding box
        refined_logits_volume[(slice(None), bbox[0], bbox[1], bbox[2])] = refined_logits_unpadded
        
        # 8. Get the final refined mask
        refined_mask = np.argmax(refined_logits_volume, axis=0).astype(np.int16)
        
    print(f"   -> Refinement complete. Mask shape: {refined_mask.shape}")
    
    # --- PHASE 3: POST-PROCESSING (The "Pink Box") ---
    print("--- PHASE 3: Post-Processing ---")
    
    # Let's do a simple per-class cleanup
    final_clean_mask = np.zeros_like(refined_mask)
    for class_id in range(1, NUM_REFINER_CLASSES): # Skip background
        class_mask = (refined_mask == class_id)
        cleaned_class_mask = post_process_segmentation(
            class_mask, 
            min_size=args.min_object_size, 
            smooth_iterations=2
        )
        final_clean_mask[cleaned_class_mask > 0] = class_id
        
    print("   -> Post-processing complete.")

    # --- PHASE 4: SAVE OUTPUT ---
    print("--- PHASE 4: Saving Final NIfTI ---")
    
    print(f"Resampling final mask back to original shape: {volume.shape}...")
    final_mask_tensor = torch.from_numpy(final_clean_mask).float().unsqueeze(0).unsqueeze(0)
    final_mask_resampled = F.interpolate(final_mask_tensor, size=volume.shape, mode='nearest')
    final_mask_highres = final_mask_resampled.squeeze().numpy().astype(np.int16)
    
    # Save the final NIfTI using the original's header and affine
    final_nii = nib.Nifti1Image(final_mask_highres, original_affine, original_header)
    nib.save(final_nii, args.output_nii)
    
    print("=" * 50)
    print(f"✅ Success! Final segmentation saved to: {args.output_nii}")
    print("=" * 50)

if __name__ == "__main__":
    main()
