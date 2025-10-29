import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects
import glob
import os
import argparse
from tqdm import tqdm

# A fixed patch size for training. All patches will be cropped/padded to this.
# (96, 96, 96) is a common size for 3D medical patches. Adjust as needed.
PATCH_SIZE = (96, 96, 96)

# --- 1. The 3D CNN Model ---
# As we discussed, a full 3D U-Net is better for segmentation.
# But your Simple3DRefiner is a great, fast starting point.

class Simple3DRefiner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # This stack processes the input at its original resolution
        self.conv_stack = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # 1x1 conv to map features to output classes
            nn.Conv3d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x is shape [B, C, D, H, W]
        return self.conv_stack(x)

# --- 2. The Custom Dataset ---
class RefinementDataset(Dataset):
    def __init__(self, ct_files, gnn_prob_files, true_label_files, patch_size):
        self.ct_files = ct_files
        self.gnn_prob_files = gnn_prob_files
        self.true_label_files = true_label_files
        self.patch_size = np.array(patch_size)
        print(f"Found {len(self.ct_files)} matching file sets for the dataset.")

    def __len__(self):
        return len(self.ct_files)

    def pad_or_crop_to_size(self, data, target_shape):
        """
        Pads or crops a 3D/4D volume to a target shape.
        data can be shape [D, H, W] or [C, D, H, W]
        """
        current_shape = np.array(data.shape)
        
        # 3D (mask) or 4D (input)
        spatial_dims = current_shape[-3:]
        target_spatial_shape = target_shape

        # Calculate padding/cropping for spatial dimensions
        shape_diff = target_spatial_shape - spatial_dims
        
        # --- Cropping (if data is too large) ---
        crop_pre = np.maximum(0, -shape_diff) // 2
        crop_post = np.maximum(0, -shape_diff) - crop_pre
        
        if data.ndim == 4: # 4D Input
            slices = (slice(None), # Keep all channels
                      slice(crop_pre[0], current_shape[1] - crop_post[0]),
                      slice(crop_pre[1], current_shape[2] - crop_post[1]),
                      slice(crop_pre[2], current_shape[3] - crop_post[2]))
        else: # 3D Mask
            slices = (slice(crop_pre[0], current_shape[0] - crop_post[0]),
                      slice(crop_pre[1], current_shape[1] - crop_post[1]),
                      slice(crop_pre[2], current_shape[2] - crop_post[2]))
        
        data = data[slices]
        
        # --- Padding (if data is too small) ---
        pad_pre = np.maximum(0, shape_diff) // 2
        pad_post = np.maximum(0, shape_diff) - pad_pre
        
        if data.ndim == 4: # 4D Input
            pad_width = ((0, 0), # No padding for channels
                         (pad_pre[0], pad_post[0]),
                         (pad_pre[1], pad_post[1]),
                         (pad_pre[2], pad_post[2]))
        else: # 3D Mask
            pad_width = ((pad_pre[0], pad_post[0]),
                         (pad_pre[1], pad_post[1]),
                         (pad_pre[2], pad_post[2]))

        data = np.pad(data, pad_width, mode='constant', constant_values=0)
        
        return data

    def __getitem__(self, idx):
        # 1. Load all three volumes
        ct_vol = nib.load(self.ct_files[idx]).get_fdata(dtype=np.float32)
        gnn_prob_vol = nib.load(self.gnn_prob_files[idx]).get_fdata(dtype=np.float32)
        true_label_vol = nib.load(self.true_label_files[idx]).get_fdata(dtype=np.int16)

        # 2. Find bounding box from the *true label*
        # Using the true label is more stable for training.
        bbox_slices = find_objects(true_label_vol > 0)
        if not bbox_slices:
            # Handle empty mask, return a standard empty patch
            num_gnn_channels = gnn_prob_vol.shape[-1]
            num_in_channels = 1 + num_gnn_channels
            return (
                torch.zeros(num_in_channels, *self.patch_size, dtype=torch.float32),
                torch.zeros(*self.patch_size, dtype=torch.long)
            )
        
        slices = bbox_slices[0]
        
        # 3. Crop all three volumes to the bounding box
        ct_patch = ct_vol[slices]
        gnn_prob_patch = gnn_prob_vol[slices]
        true_label_patch = true_label_vol[slices]
        
        # 4. Stack inputs
        # gnn_prob_patch is [D, H, W, C_gnn]. Move C_gnn to the front -> [C_gnn, D, H, W]
        gnn_prob_patch = np.moveaxis(gnn_prob_patch, -1, 0)
        # ct_patch is [D, H, W]. Add channel dim -> [1, D, H, W]
        ct_patch = np.expand_dims(ct_patch, axis=0)
        
        # Stack CT and GNN probs as channels
        input_patch = np.concatenate([ct_patch, gnn_prob_patch], axis=0)
        
        # 5. Pad or crop to fixed patch size
        input_tensor_np = self.pad_or_crop_to_size(input_patch, self.patch_size)
        target_tensor_np = self.pad_or_crop_to_size(true_label_patch, self.patch_size)

        # 6. Convert to Tensors
        input_tensor = torch.from_numpy(input_tensor_np.copy())
        target_tensor = torch.from_numpy(target_tensor_np.copy()).long()
        
        return input_tensor, target_tensor

# --- 3. The Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train a 3D CNN Refinement Model.")
    # --- Data Arguments ---
    parser.add_argument('--ct_dir', type=str, required=True, help="Directory containing original CT scans (*.nii.gz)")
    parser.add_argument('--gnn_prob_dir', type=str, required=True, help="Directory containing GNN probability maps (*_prob_map.nii.gz)")
    parser.add_argument('--label_dir', type=str, required=True, help="Directory containing true label masks (*.nii.gz)")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data to use for validation (e.g., 0.2 for 20%)")
    
    # --- Model Arguments ---
    # GNN output 5 classes, so prob map has 5 channels. 1 (CT) + 5 (GNN) = 6
    parser.add_argument('--in_channels', type=int, default=6, help="Input channels (1 for CT + num GNN classes)")
    parser.add_argument('--out_channels', type=int, default=5, help="Output classes (must match true labels)")
    
    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_model_path', type=str, default="refiner_model.pt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Find all matching files ---
    # This assumes filenames are identical across directories
    # e.g., "case_001.nii.gz" exists in all 3 folders.
    print("Finding matching data files...")
    ct_files = sorted(glob.glob(os.path.join(args.ct_dir, "*.nii.gz")))
    
    gnn_prob_files = []
    true_label_files = []
    valid_ct_files = []
    
    for ct_path in ct_files:
        base_name = os.path.basename(ct_path)
        
        # We need to find the matching _prob_map.nii.gz
        # This is tricky. Let's assume the GNN output folder is named after the case
        # e.g., ct_path = 'data/CT/case_001.nii.gz'
        # gnn_prob_path = 'data/GNN_output/case_001/final_segmentation_prob_map.nii.gz'
        # label_path = 'data/Labels/case_001.nii.gz'
        # This is too complex. Let's simplify and assume the prob maps are
        # also named 'case_001.nii.gz' or 'case_001_prob_map.nii.gz'
        
        # --- This file-matching logic is CRITICAL ---
        # --- You MUST adapt it to your file naming convention ---
        
        # Let's assume:
        # ct_dir/case_001.nii.gz
        # label_dir/case_001.nii.gz
        # gnn_prob_dir/case_001_prob_map.nii.gz  (This is what postprocess.py creates)
        
        label_path = os.path.join(args.label_dir, base_name)
        
        # Let's assume the prob map is named based on the output of central.py
        # e.g., if input is hippocampus_001.nii.gz, output is final_segmentation_prob_map.nii.gz
        # This means the gnn_prob_dir needs to be the *run* folder, e.g., 'output/run_001'
        # And the names won't match.
        
        # --- A much simpler file-finding logic ---
        # Let's assume all files *in* the directories match by name
        # e.g. ct_dir/001.nii.gz, gnn_prob_dir/001.nii.gz, label_dir/001.nii.gz
        
        label_path = os.path.join(args.label_dir, base_name)
        gnn_prob_path = os.path.join(args.gnn_prob_dir, base_name.replace(".nii.gz", "_prob_map.nii.gz")) # Assuming this pattern
        
        if os.path.exists(gnn_prob_path) and os.path.exists(label_path):
            valid_ct_files.append(ct_path)
            gnn_prob_files.append(gnn_prob_path)
            true_label_files.append(label_path)
        else:
            print(f"Warning: Skipping {base_name}, missing GNN or Label file.")
            
    # --- 2. Create Datasets and DataLoaders ---
    # Split into train and validation
    num_files = len(valid_ct_files)
    val_size = int(num_files * args.val_split)
    train_size = num_files - val_size
    
    indices = np.random.permutation(num_files)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    def get_files_from_indices(file_list, indices):
        return [file_list[i] for i in indices]

    train_dataset = RefinementDataset(
        get_files_from_indices(valid_ct_files, train_indices),
        get_files_from_indices(gnn_prob_files, train_indices),
        get_files_from_indices(true_label_files, train_indices),
        patch_size=PATCH_SIZE
    )
    val_dataset = RefinementDataset(
        get_files_from_indices(valid_ct_files, val_indices),
        get_files_from_indices(gnn_prob_files, val_indices),
        get_files_from_indices(true_label_files, val_indices),
        patch_size=PATCH_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # --- 3. Init Model, Loss, Optimizer ---
    model = Simple3DRefiner(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # --- 5. Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output_model_path)
            print(f"   -> New best model saved to {args.output_model_path} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best model saved to {args.output_model_path}")

if __name__ == "__main__":
    main()
