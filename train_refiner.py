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
    def __init__(self, ct_files, gnn_prob_files, true_label_files, patch_size, max_classes):
        self.ct_files = ct_files
        self.gnn_prob_files = gnn_prob_files
        self.true_label_files = true_label_files
        self.patch_size = np.array(patch_size)
        self.max_classes = max_classes # This is the target number of output classes
        print(f"Found {len(self.ct_files)} matching file sets for the dataset.")

    def __len__(self):
        return len(self.ct_files)

    def pad_or_crop_to_size(self, data, target_shape):
        # ... (This helper function is unchanged) ...
        current_shape = np.array(data.shape)
        spatial_dims = current_shape[-3:]
        target_spatial_shape = target_shape
        shape_diff = target_spatial_shape - spatial_dims
        crop_pre = np.maximum(0, -shape_diff) // 2
        crop_post = np.maximum(0, -shape_diff) - crop_pre
        
        if data.ndim == 4: # 4D Input
            slices = (slice(None), 
                      slice(crop_pre[0], current_shape[1] - crop_post[0]),
                      slice(crop_pre[1], current_shape[2] - crop_post[1]),
                      slice(crop_pre[2], current_shape[3] - crop_post[2]))
        else: # 3D Mask
            slices = (slice(crop_pre[0], current_shape[0] - crop_post[0]),
                      slice(crop_pre[1], current_shape[1] - crop_post[1]),
                      slice(crop_pre[2], current_shape[2] - crop_post[2]))
        
        data = data[slices]
        pad_pre = np.maximum(0, shape_diff) // 2
        pad_post = np.maximum(0, shape_diff) - pad_pre
        
        if data.ndim == 4: # 4D Input
            pad_width = ((0, 0), (pad_pre[0], pad_post[0]),
                         (pad_pre[1], pad_post[1]), (pad_pre[2], pad_post[2]))
        else: # 3D Mask
            pad_width = ((pad_pre[0], pad_post[0]),
                         (pad_pre[1], pad_post[1]), (pad_pre[2], pad_post[2]))
        
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
        return data

    def __getitem__(self, idx):
        # 1. Load all three volumes
        ct_vol_hi_res = nib.load(self.ct_files[idx]).get_fdata(dtype=np.float32)
        gnn_prob_vol = nib.load(self.gnn_prob_files[idx]).get_fdata(dtype=np.float32)
        true_label_vol_hi_res_float = nib.load(self.true_label_files[idx]).get_fdata()
        true_label_vol_hi_res = true_label_vol_hi_res_float.astype(np.int16)
        
        # --- Resample high-res volumes to low-res GNN space ---
        target_spatial_shape = gnn_prob_vol.shape[:-1] # All dims except channel
        
        # Resample CT
        ct_tensor = torch.from_numpy(ct_vol_hi_res).float().unsqueeze(0).unsqueeze(0)
        ct_resampled = F.interpolate(ct_tensor, size=target_spatial_shape, mode='trilinear', align_corners=False)
        ct_vol = ct_resampled.squeeze().numpy()
        
        # Resample Label
        label_tensor = torch.from_numpy(true_label_vol_hi_res).float().unsqueeze(0).unsqueeze(0)
        label_resampled = F.interpolate(label_tensor, size=target_spatial_shape, mode='nearest')
        true_label_vol = label_resampled.squeeze().numpy().astype(np.int16)
        
        # --- ❗ THIS IS THE FIX ---
        # We must clamp the labels. The model only outputs 'max_classes' (e.g., 5).
        # Any ground-truth label >= 5 (e.g., 6, 7... 11) is invalid
        # and must be re-mapped, usually to background (0).
        true_label_vol[true_label_vol >= self.max_classes] = 0
        # --- END FIX ---
        
        # --- Pad or Truncate GNN prob map to have 'max_classes' channels ---
        current_gnn_channels = gnn_prob_vol.shape[-1]
        target_channels = self.max_classes
        
        if current_gnn_channels < target_channels:
            pad_width_channels = target_channels - current_gnn_channels
            pad_dims = [(0, 0)] * gnn_prob_vol.ndim
            pad_dims[-1] = (0, pad_width_channels)
            gnn_prob_vol = np.pad(gnn_prob_vol, pad_dims, 'constant', constant_values=0)
        
        elif current_gnn_channels > target_channels:
            gnn_prob_vol = gnn_prob_vol[..., :target_channels]

        # 2. Find bounding box from the *resampled* true label
        bbox_slices = find_objects(true_label_vol > 0)
        if not bbox_slices:
            num_in_channels = 1 + self.max_classes
            return (
                torch.zeros(num_in_channels, *self.patch_size, dtype=torch.float32),
                torch.zeros(*self.patch_size, dtype=torch.long)
            )
        
        slices = bbox_slices[0]
        
        # 3. Crop all three (now-matching) volumes
        ct_patch = ct_vol[slices]
        gnn_prob_patch = gnn_prob_vol[slices]
        true_label_patch = true_label_vol[slices]
        
        # 4. Stack inputs
        gnn_prob_patch = np.moveaxis(gnn_prob_patch, -1, 0) # [D,H,W,C] -> [C,D,H,W]
        ct_patch = np.expand_dims(ct_patch, axis=0)         # [D,H,W]   -> [1,D,H,W]
        
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
    parser.add_argument('--in_channels', type=int, default=6, help="Input channels (1 for CT + num GNN classes)")
    parser.add_argument('--out_channels', type=int, default=5, help="Output classes (must match true labels)")
    
    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_model_path', type=str, default="refiner_model.pt")
    args = parser.parse_args()

    # --- Check channel consistency ---
    if args.in_channels != (1 + args.out_channels):
        print(f"Warning: --in_channels ({args.in_channels}) does not equal 1 + --out_channels ({args.out_channels}).")
        print(f"Setting --in_channels to {1 + args.out_channels}")
        args.in_channels = 1 + args.out_channels
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Find all matching files ---
    print("Finding matching data files...")
    ct_files = sorted(glob.glob(os.path.join(args.ct_dir, "*.nii.gz")))
    
    gnn_prob_files = []
    true_label_files = []
    valid_ct_files = []
    
    for ct_path in ct_files:
        base_name = os.path.basename(ct_path)
        
        label_path = os.path.join(args.label_dir, base_name)
        # This is the naming convention from organize_data.py
        gnn_prob_path = os.path.join(args.gnn_prob_dir, base_name.replace(".nii.gz", "_prob_map.nii.gz"))
        
        if os.path.exists(gnn_prob_path) and os.path.exists(label_path):
            valid_ct_files.append(ct_path)
            gnn_prob_files.append(gnn_prob_path)
            true_label_files.append(label_path)
        else:
            print(f"Warning: Skipping {base_name}, missing GNN or Label file.")
            
    # --- 2. Create Datasets and DataLoaders ---
    num_files = len(valid_ct_files)
    if num_files == 0:
        print("❌ FATAL: No matching files found. Check your --ct_dir, --gnn_prob_dir, and --label_dir paths.")
        return
        
    val_size = int(num_files * args.val_split)
    if num_files > 1 and val_size == 0:
        val_size = 1
    train_size = num_files - val_size
    
    if train_size == 0:
        print(f"❌ FATAL: Not enough data to train. Found {num_files} total files, but need at least 1 for training.")
        return

    indices = np.random.permutation(num_files)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    def get_files_from_indices(file_list, indices):
        return [file_list[i] for i in indices]

    # --- Pass 'args.out_channels' to the Dataset ---
    train_dataset = RefinementDataset(
        get_files_from_indices(valid_ct_files, train_indices),
        get_files_from_indices(gnn_prob_files, train_indices),
        get_files_from_indices(true_label_files, train_indices),
        patch_size=PATCH_SIZE,
        max_classes=args.out_channels # Pass the target class count
    )
    
    if val_size > 0:
        val_dataset = RefinementDataset(
            get_files_from_indices(valid_ct_files, val_indices),
            get_files_from_indices(gnn_prob_files, val_indices),
            get_files_from_indices(true_label_files, val_indices),
            patch_size=PATCH_SIZE,
            max_classes=args.out_channels # Pass the target class count
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None
        print("Warning: No validation set created. Training on all data.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
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
        if val_loader:
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
        else:
             print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
             if epoch == args.epochs - 1:
                torch.save(model.state_dict(), args.output_model_path)

    print("\n--- Training Complete ---")
    print(f"Model saved to {args.output_model_path}")

if __name__ == "__main__":
    main()
