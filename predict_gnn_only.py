# predict_gnn_only.py
"""
This is a "GNN-only" prediction script.
It loads the MASTER GNN, runs it on a single CT scan,
and saves the resulting low-res probability map.
This is used to generate the dataset for the Phase 2 refiner.
"""

import argparse
import os
import sys
import pickle
import joblib
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
from scipy import stats, ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TopKPooling

import timm
from torchvision import transforms
import SimpleITK as sitk
from skimage.segmentation import slic

# Import the GNN model class from our trainer
from train_unet import GNNUNet

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- All helper functions from predict.py ---
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

def compute_supervoxel_features(volume, labels):
    regions = np.unique(labels)
    features = []
    for label in regions:
        if label == 0: continue
        mask = (labels == label)
        if np.sum(mask) == 0: continue
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
    all_sv_ids = np.unique(supervoxels)
    all_sv_ids = all_sv_ids[all_sv_ids != 0]
    for label in all_sv_ids:
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
    labels = labels[labels != 0]
    for label in labels:
        mask = (supervoxels == label)
        dilated = ndimage.binary_dilation(mask, structure=structure)
        neighbors = np.unique(supervoxels[dilated])
        neighbors = neighbors[neighbors != label]
        neighbors = neighbors[neighbors != 0]
        adjacency[label] = neighbors.tolist()
    return adjacency
# --- End Helper Functions ---


def main():
    parser = argparse.ArgumentParser(description="GNN-Only Prediction Pipeline")
    parser.add_argument("--input_ct", type=str, required=True, help="Path to the new, unseen CT.nii.gz file.")
    parser.add_argument("--output_prob_map_nii", type=str, required=True, help="Path to save the GNN probability map.")
    parser.add_argument("--gnn_model_path", type=str, required=True, help="Path to the MASTER GNN (gnn_master_model.pt)")
    parser.add_argument("--n_segments", type=int, default=400, help="Number of supervoxels to generate")
    parser.add_argument("--num_gnn_classes", type=int, default=28, help="Number of classes the GNN model was trained to output.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    
    # --- PHASE 1A: PREPROCESSING ---
    print("--- PHASE 1A: Preprocessing ---")
    print(f"Loading {args.input_ct}...")
    volume, spacing, original_affine, original_header = load_nifti_for_predict(args.input_ct)
    print(f"   -> Original shape: {volume.shape}")

    if volume.ndim != 3:
        print(f"   -> ❌ FATAL: Input volume is not 3D. Shape is {volume.shape}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Resampling...")
    resampled_vol = resample_volume(volume, spacing)
    print(f"   -> Resampled shape: {resampled_vol.shape}")
    
    print("Normalizing...")
    preprocessed_vol = preprocess(resampled_vol)
    
    print("Computing supervoxels...")
    supervoxels_vol = compute_supervoxels(preprocessed_vol, n_segments=args.n_segments)
    print(f"   -> Supervoxel array shape: {supervoxels_vol.shape}")

    # --- PHASE 1B: GRAPH BUILDING ---
    print("--- PHASE 1B: Graph Building ---")
    print("Loading DINO model...")
    dino_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(DEVICE)
    dino_model.eval()
    dino_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
    ])

    print("Extracting DINO features...")
    all_sv_ids = np.unique(supervoxels_vol)
    all_sv_ids = all_sv_ids[all_sv_ids != 0]
    
    dino_features_list = []
    for sv_id in tqdm(all_sv_ids, desc="DINO Features"):
        patch = _get_supervoxel_2d_patch(preprocessed_vol, supervoxels_vol, sv_id)
        if patch is None or patch.max() == 0:
            dino_feat = np.zeros(768)
        else:
            dino_feat = _extract_dino_feature_2d(patch, dino_model, dino_transform)
        dino_features_list.append(dino_feat)
    
    dino_features_array = np.vstack(dino_features_list)
    dino_df = pd.DataFrame(dino_features_array, index=all_sv_ids).add_prefix('dino_feat_')
    
    print("Extracting spatial & intensity features...")
    spatial_df = _compute_spatial_features(supervoxels_vol).set_index('supervoxel_id')
    intensity_df = compute_supervoxel_features(preprocessed_vol, supervoxels_vol).set_index('supervoxel_id')

    print("Combining features...")
    features_df = spatial_df.join(dino_df).join(intensity_df).fillna(0)
    print(f"   -> Final feature vector shape: {features_df.shape}")
    
    print("Computing adjacency...")
    adjacency = _get_adjacency(supervoxels_vol)
    
    print("Building NetworkX graph...")
    G = nx.Graph()
    feat_dict = {sv_id: row.values for sv_id, row in features_df.iterrows()}
    
    for sv_id, features in feat_dict.items():
        G.add_node(sv_id, features=features)
        
    for node, neighbors in adjacency.items():
        if node in G.nodes(): 
            for n in neighbors:
                if n in G.nodes():
                    G.add_edge(node, n)
    
    print(f"   -> Graph built with {G.number_of_nodes()} nodes.")
    
    # --- PHASE 1C: GNN INFERENCE ---
    print("--- PHASE 1C: GNN Inference ---")
    
    node_list = sorted(list(G.nodes()))
    node_map = {node: i for i, node in enumerate(node_list)}
    inv_node_map = {i: node for node, i in node_map.items()}

    node_features = [torch.tensor(G.nodes[n]['features']) for n in node_list]
    x = torch.stack(node_features, dim=0).float().to(DEVICE)
    
    remapped_edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous().to(DEVICE)
    
    data = Data(x=x, edge_index=edge_index)

    print(f"Loading GNN model from {args.gnn_model_path}...")
    num_node_features = data.num_node_features
    NUM_GNN_CLASSES = args.num_gnn_classes 
    
    gnn_model = GNNUNet(
        in_channels=num_node_features, hidden_channels=64,
        out_channels=NUM_GNN_CLASSES, heads=4
    ).to(DEVICE)
    
    gnn_model.load_state_dict(torch.load(args.gnn_model_path, map_location=DEVICE))
    gnn_model.eval()
    
    print("Running GNN inference...")
    with torch.no_grad():
        log_probs = gnn_model(data.x, data.edge_index)
        probs = torch.exp(log_probs).cpu().numpy()

    # 4. Reproject to create Coarse Probability Map
    print("Reprojecting GNN predictions to 3D coarse map...")
    coarse_prob_map = np.zeros(supervoxels_vol.shape + (NUM_GNN_CLASSES,), dtype=np.float32)

    for model_idx in range(probs.shape[0]):
        if model_idx in inv_node_map: 
            original_sv_id = inv_node_map[model_idx]
            node_probs = probs[model_idx] 
            voxel_coords = (supervoxels_vol == original_sv_id)
            coarse_prob_map[voxel_coords] = node_probs
    
    print(f"   -> Coarse map shape: {coarse_prob_map.shape}")
    
    # --- PHASE 4: SAVE OUTPUT ---
    print("--- PHASE 4: Saving Probability Map ---")
    
    # We save this in the *resampled* (low-res) space.
    # The refiner will load this low-res map.
    # We must use an affine that matches.
    
    # Create a new affine for the resampled, 1x1x1 space
    # This is a simple identity affine, which is correct for a 1x1x1 resampled image
    resampled_affine = np.eye(4)
    
    prob_map_nii = nib.Nifti1Image(coarse_prob_map, resampled_affine)
    nib.save(prob_map_nii, args.output_prob_map_nii)
    
    print("=" * 50)
    print(f"✅ Success! Coarse probability map saved to: {args.output_prob_map_nii}")
    print("=" * 50)

if __name__ == "__main__":
    main()
