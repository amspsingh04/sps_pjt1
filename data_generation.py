# run_data_generation.py
import subprocess
import sys
import os
import glob
from tqdm import tqdm

# --- Configuration ---
NUM_FILES_TO_RUN = 10
GNN_MODEL_PATH = "gnn_master_model.pt" # The model you just trained
NUM_GNN_CLASSES = 28 # The --num_classes you used for the master GNN
N_SEGMENTS = 400 # The --n_segments you want to use for prediction

IMAGE_PATTERN = "PanTS/data/ImageTr/PanTS_*/ct.nii.gz"
OUTPUT_DATASET_DIR = "data_for_refiner_MASTER" # <-- New, clean folder

# --- Main Script ---
def main():
    print("Finding and sorting data files...")
    image_files = sorted(glob.glob(IMAGE_PATTERN))
    if not image_files:
        print(f"❌ Error: No files found for pattern '{IMAGE_PATTERN}'")
        sys.exit(1)

    # Get just the first 10 files
    files_to_process = image_files[:NUM_FILES_TO_RUN]
    print(f"Found {len(files_to_process)} file pairs to process.")
    
    # Create the output directory
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    for img_path in tqdm(files_to_process, desc="Generating Refiner Data"):
        
        case_id = os.path.basename(os.path.dirname(img_path))
        print(f"\n--- Processing Case: {case_id} ---")
        
        # Define the output name for the probability map
        output_prob_map = os.path.join(OUTPUT_DATASET_DIR, f"{case_id}_prob_map.nii.gz")

        command = [
            sys.executable, "predict_gnn_only.py",
            "--input_ct", img_path,
            "--output_prob_map_nii", output_prob_map,
            "--gnn_model_path", GNN_MODEL_PATH,
            "--num_gnn_classes", str(NUM_GNN_CLASSES),
            "--n_segments", str(N_SEGMENTS)
        ]
        
        try:
            subprocess.run(command, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {case_id}. GNN prediction failed.")
            print(f"   Command was: {' '.join(command)}")
            print(f"   Return code: {e.returncode}")
            print(f"   Stderr: {e.stderr}")
            print("   Aborting the rest of the runs.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 User interrupted the process. Exiting.")
            sys.exit(1)

    print("\n✅ Successfully generated all probability maps for the refiner.")

if __name__ == "__main__":
    main()
