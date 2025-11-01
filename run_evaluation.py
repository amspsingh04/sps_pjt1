# run_evaluation.py
import subprocess
import sys
import os
import glob
from tqdm import tqdm
import numpy as np 

# --- Configuration ---
# Point this to your GNN output folders
GNN_OUTPUT_PATTERN = "output/run_PanTS_*"

# Point this to your ground-truth TRAINING labels
LABEL_DIR = "PanTS/data/labelsTr"

# The evaluation script we want to run
EVAL_SCRIPT = "evaluate_dice.py"

# --- ❗ FIX: Set how many files you want to check ---
NUM_FILES_TO_EVAL = 10
# --- END FIX ---


# --- Main Script ---
def main():
    print(f"Finding all pipeline outputs in: {GNN_OUTPUT_PATTERN}")
    gnn_run_folders = sorted(glob.glob(GNN_OUTPUT_PATTERN))
    
    if not gnn_run_folders:
        print(f"❌ Error: No output folders found at '{GNN_OUTPUT_PATTERN}'")
        sys.exit(1)

    # --- ❗ FIX: Slice the list to get just the first 10 ---
    files_to_evaluate = gnn_run_folders[:NUM_FILES_TO_EVAL]
    print(f"Found {len(gnn_run_folders)} total runs, evaluating the first {len(files_to_evaluate)}.")
    # --- END FIX ---
    
    all_mean_dice_scores = []

    # --- ❗ FIX: Loop over the *sliced* list ---
    for run_folder in tqdm(files_to_evaluate, desc="Evaluating Cases"):
        # Get the case ID, e.g., "PanTS_00000011"
        case_id = os.path.basename(run_folder).replace("run_", "")
        
        print(f"\n──────────────────────────────────────────────────")
        print(f"--- Evaluating Case: {case_id} ---")

        pred_path = os.path.join(run_folder, "final_segmentation.nii.gz")
        label_path = os.path.join(LABEL_DIR, case_id, "combined_labels.nii.gz")

        if not os.path.exists(pred_path):
            print(f"   -> ⚠️ SKIPPING: Cannot find prediction file: {pred_path}")
            continue
        if not os.path.exists(label_path):
            print(f"   -> ⚠️ SKIPPING: Cannot find label file: {label_path}")
            continue

        command = [
            sys.executable,
            EVAL_SCRIPT,
            "--pred", pred_path,
            "--label", label_path
        ]
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"   -> ❌ FAILED to evaluate {case_id}.")
                print("   --- Captured Stderr: ---")
                print(stderr if stderr else "No stderr output.")
                print("   ------------------------")
            else:
                print(stdout)
                for line in stdout.splitlines():
                    if "Mean Dice" in line:
                        mean_dice = float(line.split()[-1])
                        all_mean_dice_scores.append(mean_dice)
                        
        except Exception as e:
            print(f"   -> ❌ PYTHON FAILED to even *run* subprocess.")
            print(e)

    # After the loop, print the final average of all runs
    if all_mean_dice_scores:
        final_average_dice = np.mean(all_mean_dice_scores)
        print(f"\n====================================================")
        print(f"📊 Final Average Dice over {len(all_mean_dice_scores)} cases: {final_average_dice:.4f}")
        print(f"====================================================")
    else:
        print("\n" + "="*50)
        print("No Dice scores were calculated. All evaluations failed or produced 0.")
        print("="*50)

if __name__ == "__main__":
    main()
