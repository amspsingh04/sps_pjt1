import argparse
import numpy as np
import pandas as pd
import joblib
import torch
from torchvision import transforms
from PIL import Image
import timm
from scipy import ndimage
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def extract_dino_feature_2d(patch_2d, model, transform):
    # patch_2d: numpy 2D grayscale uint8
    img_3ch = np.stack([patch_2d]*3, axis=-1)
    input_tensor = transform(img_3ch).unsqueeze(0)  # batch 1
    with torch.no_grad():
        feat = model.forward_features(input_tensor)
    return feat.squeeze().cpu().numpy()


def get_supervoxel_2d_patch(volume, supervoxels, sv_id):
    # Find slices where sv_id exists, pick slice with max pixels
    slices_idx = np.where(np.any(supervoxels == sv_id, axis=(1,2)))[0]
    if len(slices_idx) == 0:
        return None
    max_slice = max(slices_idx, key=lambda z: np.sum(supervoxels[z] == sv_id))
    mask_2d = (supervoxels[max_slice] == sv_id)
    coords = np.argwhere(mask_2d)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    patch = volume[max_slice, y0:y1, x0:x1]
    # Normalize patch to 0-255 uint8
    patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
    patch_img = (patch_norm * 255).astype(np.uint8)
    return patch_img


def compute_spatial_features(supervoxels):
    spatial_feats = []
    labels = np.unique(supervoxels)
    for label in labels:
        coords = np.array(np.where(supervoxels == label))
        centroid = coords.mean(axis=1)
        size = coords.ptp(axis=1)
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
    adjacency = {}
    labels = np.unique(supervoxels)
    for label in labels:
        mask = (supervoxels == label)
        dilated = ndimage.binary_dilation(mask, structure=structure)
        neighbors = np.unique(supervoxels[dilated])
        neighbors = neighbors[neighbors != label]
        adjacency[label] = neighbors.tolist()
    return adjacency


def main():
    parser = argparse.ArgumentParser(description="Build graph from supervoxels using DINO features")
    parser.add_argument("volume_pkl", type=str, help="Path to preprocessed volume .pkl")
    parser.add_argument("supervoxels_pkl", type=str, help="Path to supervoxels .pkl")
    parser.add_argument("features_csv", type=str, help="Path to intensity features .csv")
    parser.add_argument("--out_graph", type=str, default="output/supervoxel_graph.gpickle", help="Output graph path")

    args = parser.parse_args()

    print("Loading data...")
    volume = joblib.load(args.volume_pkl)
    supervoxels = joblib.load(args.supervoxels_pkl)
    features_df = pd.read_csv(args.features_csv)

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
        patch = get_supervoxel_2d_patch(volume, supervoxels, sv_id)
        if patch is None:
            dino_feat = np.zeros(768)
        else:
            dino_feat = extract_dino_feature_2d(patch, model, transform)
        dino_features_list.append(dino_feat)

    dino_features_array = np.vstack(dino_features_list)
    dino_df = pd.DataFrame(dino_features_array, index=features_df.index).add_prefix('dino_feat_')

    print("Computing spatial features...")
    spatial_df = compute_spatial_features(supervoxels)

    print("Combining all features...")
    all_feats = features_df.merge(spatial_df, on='supervoxel_id').reset_index(drop=True)
    all_feats = pd.concat([all_feats, dino_df.reset_index(drop=True)], axis=1)

    print("Computing adjacency...")
    adjacency = get_adjacency(supervoxels)

    print("Computing edge weights and building graph...")
    feat_dict = {row['supervoxel_id']: row.drop('supervoxel_id').values for _, row in all_feats.iterrows()}

    edges = []
    for node, neighbors in adjacency.items():
        feat1 = feat_dict[node].reshape(1, -1)
        for n in neighbors:
            feat2 = feat_dict[n].reshape(1, -1)
            weight = cosine_similarity(feat1, feat2)[0][0]
            edges.append((node, n, weight))

    G = nx.Graph()
    for node in feat_dict.keys():
        G.add_node(node, features=feat_dict[node])
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    nx.write_gpickle(G, args.out_graph)
    print(f"Graph saved to {args.out_graph}")


if __name__ == "__main__":
    main()
