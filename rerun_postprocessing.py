# rerun_postprocessing.py
import subprocess
import sys
import os
import glob
from tqdm import tqdm

# --- Configuration ---
# Point this to your GNN output folders
GNN_OUTPUT_PATTERN = "output/run_PanTS_*"

# Point this to your ORIGINAL TRAINING images (to get headers/shapes)
IMAGE_DIR = "PanTS/data/ImageTr"

# Point this to your ground-truth TRAINING labels
LABEL_DIR = "PanTS/data/labelsTr"

# The post-processing script we want to run
POST_SCRIPT = "postprocess.py"

# --- Main Script ---
def main():
    print(f"Finding all pipeline outputs in: {GNN_OUTPUT_PATTERN}")
    gnn_run_folders = sorted(glob.glob(GNN_OUTPUT_PATTERN))
    
    if not gnn_run_folders:
        print(f"❌ Error: No output folders found at '{GNN_OUTPUT_PATTERN}'")
        sys.exit(1)

    print(f"Found {len(gnn_run_folders)} folders to re-process.")

    for run_folder in tqdm(gnn_run_folders, desc="Fixing Files"):
        # Get the case ID, e.g., "PanTS_00000001"
        case_id = os.path.basename(run_folder).replace("run_", "")
        
        print(f"\n──────────────────────────────────────────────────")
        print(f"--- Fixing Case: {case_id} ---")

        # --- Define all 5 required paths ---
        supervoxels_path = os.path.join(run_folder, "preprocessed_supervoxels.pkl")
        predictions_path = os.path.join(run_folder, "node_predictions.pt")
        node_map_path = os.path.join(run_folder, "node_mapping.pkl")
        
        # This is the original, high-res CT image
        original_image_path = os.path.join(IMAGE_DIR, case_id, "ct.nii.gz")
        
        # This is the final output file we want to OVERWRITE
        output_path = os.path.join(run_folder, "final_segmentation.nii.gz")

        # Check if all inputs exist
        inputs_exist = True
        for p in [supervoxels_path, predictions_path, node_map_path, original_image_path]:
            if not os.path.exists(p):
                print(f"   -> ⚠️ SKIPPING: Cannot find input file: {p}")
                inputs_exist = False
                break
        
        if not inputs_exist:
            continue

        # --- Build and run the command ---
        command = [
            sys.executable,
            POST_SCRIPT,
            "--supervoxels_path", supervoxels_path,
            "--predictions_path", predictions_path,
            "--node_map_path", node_map_path,
            "--original_image_nii", original_image_path,
            "--output_nii", output_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"   -> ✅ Successfully fixed {case_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"   -> ❌ FAILED to fix {case_id}.")
            print("   --- Captured Stderr: ---")
            print(e.stderr if e.stderr else "No stderr output.")
            print("   ------------------------")
            
    print("\n====================================================")
    print(f"✅ All {len(gnn_run_folders)} folders have been re-processed.")
    print("====================================================")

if __name__ == "__main__":
    main()
