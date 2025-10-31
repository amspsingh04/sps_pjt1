# organize_data.py
import os
import glob
import shutil
from tqdm import tqdm

# --- Configuration ---
# 1. Where your original data lives
CT_DIR_PATTERN = "PanTS/data/ImageTr/PanTS_*"
LABEL_DIR_PATTERN = "PanTS/data/labelsTr/PanTS_*"

# 2. Where your GNN outputs are
GNN_OUTPUT_PATTERN = "output/run_PanTS_*"

# 3. Where you want the new, clean dataset to go
OUTPUT_DIR = "data_for_refiner"

# --- Main Script ---
def main():
    print("Organizing data for the Phase 2 Refiner...")
    
    # 1. Create the new directories
    ct_out = os.path.join(OUTPUT_DIR, "ct")
    labels_out = os.path.join(OUTPUT_DIR, "labels")
    gnn_probs_out = os.path.join(OUTPUT_DIR, "gnn_probs")
    
    os.makedirs(ct_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(gnn_probs_out, exist_ok=True)

    # 2. Find all the GNN output folders
    gnn_run_folders = sorted(glob.glob(GNN_OUTPUT_PATTERN))
    
    if not gnn_run_folders:
        print(f"❌ Error: No GNN output folders found at '{GNN_OUTPUT_PATTERN}'")
        return

    print(f"Found {len(gnn_run_folders)} GNN output runs to process.")

    # 3. Loop, copy, and rename
    for run_folder in tqdm(gnn_run_folders, desc="Organizing files"):
        case_id = os.path.basename(run_folder).replace("run_", "") # e.g., "PanTS_00000001"
        
        # --- Define source paths ---
        src_ct_path = os.path.join("PanTS/data/ImageTr", case_id, "ct.nii.gz")
        src_label_path = os.path.join("PanTS/data/labelsTr", case_id, "combined_labels.nii.gz")
        src_gnn_prob_path = os.path.join(run_folder, "final_segmentation_prob_map.nii.gz")
        
        # --- Define destination paths ---
        # This renaming is CRITICAL for train_refiner.py's file matching
        dest_ct_path = os.path.join(ct_out, f"{case_id}.nii.gz")
        dest_label_path = os.path.join(labels_out, f"{case_id}.nii.gz")
        dest_gnn_prob_path = os.path.join(gnn_probs_out, f"{case_id}_prob_map.nii.gz")
        
        # 4. Copy the files
        if os.path.exists(src_ct_path) and os.path.exists(src_label_path) and os.path.exists(src_gnn_prob_path):
            shutil.copy(src_ct_path, dest_ct_path)
            shutil.copy(src_label_path, dest_label_path)
            shutil.copy(src_gnn_prob_path, dest_gnn_prob_path)
        else:
            print(f"\nWarning: Skipping {case_id}, a source file was missing.")
            
    print(f"\n✅ Done. Your refiner dataset is ready in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
