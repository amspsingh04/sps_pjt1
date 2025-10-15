import argparse
import numpy as np
import pandas as pd
import joblib
import torch
from torchvision import transforms
import timm
from scipy import ndimage
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def extract_dino_feature_2d(patch_2d, model, transform):

    # Convert single channel to 3-channel by stacking
    img_3ch = np.stack([patch_2d]*3, axis=-1).astype(np.uint8)
    input_tensor = transform(img_3ch).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        feat = model.forward_features(input_tensor)

    return feat.squeeze().cpu().numpy()


def get_supervoxel_2d_patch(volume, supervoxels, sv_id, patch_size=96):

    slices_idx = np.where(np.any(supervoxels == sv_id, axis=(1,2)))[0]
    if len(slices_idx) == 0:
        raise ValueError(f"Supervoxel {sv_id} not found in any slice")

    slice_idx = slices_idx[len(slices_idx)//2]

    mask_2d = (supervoxels[slice_idx] == sv_id)  # shape (Y,X)

    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        raise ValueError(f"Supervoxel {sv_id} mask empty in slice {slice_idx}")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    half = patch_size // 2
    start_y = max(center_y - half, 0)
    end_y = start_y + patch_size
    if end_y > volume.shape[1]:
        end_y = volume.shape[1]
        start_y = end_y - patch_size
    start_x = max(center_x - half, 0)
    end_x = start_x + patch_size
    if end_x > volume.shape[2]:
        end_x = volume.shape[2]
        start_x = end_x - patch_size

    patch = volume[slice_idx, start_y:end_y, start_x:end_x]

    pad_y = patch_size - patch.shape[0]
    pad_x = patch_size - patch.shape[1]

    if pad_y > 0 or pad_x > 0:
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

    return patch


def compute_spatial_features(supervoxels):
    labels = np.unique(supervoxels)
    spatial_feats = []
    for label in labels:
        coords = np.array(np.where(supervoxels == label))
        centroid = coords.mean(axis=1)
        size = coords.ptp(axis=1)  # peak to peak (max-min)
        spatial_feats.append({
            "supervoxel_id": label,
            "centroid_z": centroid[0],
            "centroid_y": centroid[1],
            "centroid_x": centroid[2],
            "size_z": size[0],
            "size_y": size[1],
            "size_x": size[2],
        })
    return pd.DataFrame(spatial_feats)


def get_adjacency(supervoxels):
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels = np.unique(supervoxels)
    adjacency = {label: set() for label in labels}

    for label in labels:
        mask = (supervoxels == label)
        dilated = ndimage.binary_dilation(mask, structure=structure)
        neighbor_labels = np.unique(supervoxels[dilated])
        neighbor_labels = neighbor_labels[neighbor_labels != label]
        adjacency[label].update(neighbor_labels.tolist())

    adjacency = {k: list(v) for k, v in adjacency.items()}
    return adjacency


def main():
    parser = argparse.ArgumentParser(description="Build graph from supervoxels using DINO features")
    parser.add_argument("volume_pkl", type=str, help="Path to preprocessed volume .pkl")
    parser.add_argument("supervoxels_pkl", type=str, help="Path to supervoxels .pkl")
    parser.add_argument("features_csv", type=str, help="Path to intensity features .csv")
    parser.add_argument("--out_graph", type=str, default="output/supervoxel_graph.gpickle", help="Output graph path")

    args = parser.parse_args()

    print("Loading data...")
    volume = joblib.load(args.volume_pkl)  # shape (Z,Y,X)
    supervoxels = joblib.load(args.supervoxels_pkl)
    features_df = pd.read_csv(args.features_csv)

    print(f"Volume shape: {volume.shape}")
    print(f"Supervoxels shape: {supervoxels.shape}")
    assert volume.shape == supervoxels.shape, "Volume and supervoxels must have the same shape!"

    print("Loading DINO ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485]*3, std=[0.229]*3),
    ])

    print("Extracting DINO features for supervoxels...")
    dino_features_list = []
    for sv_id in features_df['supervoxel_id']:
        try:
            patch = get_supervoxel_2d_patch(volume, supervoxels, sv_id)
            dino_feat = extract_dino_feature_2d(patch, model, transform)
        except Exception as e:
            print(f"Warning: Could not extract patch for supervoxel {sv_id}: {e}")
            dino_feat = np.zeros(768)
        dino_features_list.append(dino_feat)

    dino_features_array = np.vstack(dino_features_list)
    dino_df = pd.DataFrame(dino_features_array, columns=[f'dino_feat_{i}' for i in range(dino_features_array.shape[1])])
    dino_df['supervoxel_id'] = features_df['supervoxel_id'].values

    print("Computing spatial features...")
    spatial_df = compute_spatial_features(supervoxels)

    print("Combining all features...")
    all_feats = features_df.merge(spatial_df, on='supervoxel_id')
    all_feats = all_feats.merge(dino_df, on='supervoxel_id')

    print("Computing adjacency...")
    adjacency = get_adjacency(supervoxels)

    print("Building graph with edge weights...")
    feat_dict = {row['supervoxel_id']: row.drop('supervoxel_id').values for _, row in all_feats.iterrows()}

    G = nx.Graph()
    for node in feat_dict.keys():
        G.add_node(node, features=feat_dict[node])

    for node, neighbors in adjacency.items():
        feat1 = feat_dict[node].reshape(1, -1)
        for n in neighbors:
            feat2 = feat_dict[n].reshape(1, -1)
            weight = cosine_similarity(feat1, feat2)[0][0]
            G.add_edge(node, n, weight=weight)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    import os
    out_dir = os.path.dirname(args.out_graph)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nx.write_gpickle(G, args.out_graph)
    print(f"Graph saved to {args.out_graph}")


if __name__ == "__main__":
    main()
