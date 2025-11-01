# train_unet_master.py
"""
Trains ONE Master GNN Model on ALL available graph data.

This script replaces the single-file training in 'central.py'
and is the correct "Path B" approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  # ❗ Note: This is from PyG
from torch_geometric.data import Data

import argparse
import pickle
import numpy as np
import networkx as nx
import glob
import os
from tqdm import tqdm

from train_unet import GNNUNet # <-- We import our model from the original file

# --- ❗ MODIFIED ---
# We'll hardcode this for simplicity, just like in our other scripts
NUM_FILES_TO_RUN = 100 
# --- END MODIFICATION ---

# --- 1. The Custom Dataset ---
class GraphDataset(Dataset):
    """
    A PyTorch Dataset to load our pre-processed graphs one by one.
    """
    def __init__(self, data_run_folders, max_classes):
        self.max_classes = max_classes
        self.graph_files = []
        self.label_files = []
        
        print(f"Scanning {len(data_run_folders)} folders for data...")
        for run_folder in data_run_folders:
            graph_path = os.path.join(run_folder, "supervoxel_graph.gpickle")
            label_path = os.path.join(run_folder, "supervoxel_label_mapping.pkl")
            
            if os.path.exists(graph_path) and os.path.exists(label_path):
                self.graph_files.append(graph_path)
                self.label_files.append(label_path)
        print(f"   -> Found {len(self.graph_files)} valid data pairs.")

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        # 1. Load the two files for this case
        with open(self.graph_files[idx], "rb") as f:
            G = pickle.load(f)
        with open(self.label_files[idx], 'rb') as f:
            supervoxel_labels = pickle.load(f)

        # 2. Perform all the same remapping logic from train_unet.py
        node_list = sorted(list(G.nodes()))
        node_map = {node: i for i, node in enumerate(node_list)}

        # 3. Build x (features)
        node_features = [torch.tensor(G.nodes[n]['features']) for n in node_list]
        x = torch.stack(node_features, dim=0).float()
        
        # 4. Build edge_index
        remapped_edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
        edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
        
        # 5. Build y (labels)
        original_labels = [supervoxel_labels.get(original_id, 0) for original_id in node_list]
        
        unique_original_labels = sorted(np.unique(original_labels))
        label_remap = {original_val: new_val for new_val, original_val in enumerate(unique_original_labels)}
        
        remapped_labels = [label_remap[l] for l in original_labels]
        y = torch.tensor(remapped_labels, dtype=torch.long)
        
        # --- ❗ CRITICAL CLAMPING STEP ---
        # We must ensure all labels are < max_classes
        y[y >= self.max_classes] = 0
        # ---
        
        # 6. Create the PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_list))
        return data

# --- 2. The Main Training Function ---
def main():
    parser = argparse.ArgumentParser(description="Train a Master GNN Model.")
    parser.add_argument('--data_dir_pattern', type=str, default="output/run_PanTS_*", help="Glob pattern to find all run folders.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data for validation.")
    parser.add_argument('--num_classes', type=int, default=28, help="The MAXIMUM number of classes across all datasets (e.g., 27 classes + 1 background).")
    
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8, help="How many GRAPHS to load per batch.")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_model_path', type=str, default="gnn_master_model.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Find and split data ---
    all_run_folders = sorted(glob.glob(args.data_dir_pattern))
    if not all_run_folders:
        print(f"❌ FATAL: No data folders found at '{args.data_dir_pattern}'")
        return
        
    # --- ❗ MODIFIED ---
    # Slice the list to get just the first 10
    all_run_folders = all_run_folders[:NUM_FILES_TO_RUN]
    print(f"--- ⚠️  Limiting run to first {len(all_run_folders)} files for testing. ---")
    # --- END MODIFICATION ---

    val_size = int(len(all_run_folders) * args.val_split)
    # Ensure at least 1 file for validation if we have > 1 file
    if len(all_run_folders) > 1 and val_size == 0:
        val_size = 1
        
    train_size = len(all_run_folders) - val_size
    
    if train_size == 0:
        print(f"❌ FATAL: Not enough data to train. Found {len(all_run_folders)} total files, but need at least 1 for training.")
        return

    indices = np.random.permutation(len(all_run_folders))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_folders = [all_run_folders[i] for i in train_indices]
    val_folders = [all_run_folders[i] for i in val_indices]

    # --- 2. Create Datasets and DataLoaders ---
    # We must pass the *maximum* number of classes
    train_dataset = GraphDataset(train_folders, max_classes=args.num_classes)
    val_dataset = GraphDataset(val_folders, max_classes=args.num_classes)
    
    # Check if datasets are empty (can happen if sub-files are missing)
    if len(train_dataset) == 0:
        print(f"❌ FATAL: No valid data pairs found in the training folders.")
        return

    # Use the PyG DataLoader, which knows how to batch different-sized graphs
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        print("⚠️ Warning: No validation data. Training on all 10 files.")
        val_loader = None
    
    # --- 3. Init Model, Loss, Optimizer ---
    # We need to know the number of features. Let's peek at the first graph.
    temp_data = train_dataset[0]
    num_node_features = temp_data.num_node_features
    print(f"Detected {num_node_features} node features.")
    
    model = GNNUNet(
        in_channels=num_node_features,
        hidden_channels=64, # Standard
        out_channels=args.num_classes, # Master model outputs all classes
        heads=4 # Standard
    ).to(device)
    
    criterion = nn.NLLLoss() # Good for log_softmax output
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get logits. Note: batch.y contains labels for *all nodes in the batch*
            out = model(batch.x, batch.edge_index)
            
            # We don't use masks here, we train on all nodes
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # --- 5. Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.output_model_path)
                print(f"   -> New best model saved to {args.output_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            # No validation, just train and save at the end
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")

    print("\n--- Training Complete ---")
    # If no validation, save the final model
    if not val_loader:
        torch.save(model.state_dict(), args.output_model_path)
        
    print(f"Best master GNN model saved to {args.output_model_path}")

if __name__ == "__main__":
    main()
