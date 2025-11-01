import subprocess
import sys
import os
import glob
from tqdm import tqdm # For a nice progress bar, pip install tqdm

# --- Configuration ---
NUM_FILES_TO_RUN = 100
EPOCHS = 300
LEARNING_RATE = 7e-4

# These glob patterns find ALL your files.
# Make sure these paths are correct relative to where you run this script.
IMAGE_PATTERN = "PanTS/data/ImageTr/PanTS_*/ct.nii.gz"
LABEL_PATTERN = "PanTS/data/labelsTr/PanTS_*/combined_labels.nii.gz"

# --- Main Script ---
def main():
    print("Finding and sorting data files...")
    
    # Find all files and sort them to ensure they match
    image_files = sorted(glob.glob(IMAGE_PATTERN))
    label_files = sorted(glob.glob(LABEL_PATTERN))

    # Basic check to make sure the file lists match
    if len(image_files) != len(label_files):
        print(f"❌ Error: Found {len(image_files)} images but {len(label_files)} labels.")
        print("Please check your file patterns.")
        sys.exit(1)

    if not image_files:
        print(f"❌ Error: No files found for pattern '{IMAGE_PATTERN}'")
        print("Are you in the right directory? Do the files exist?")
        sys.exit(1)

    # Get just the first 10 files
    files_to_process = list(zip(image_files, label_files))[60:60+NUM_FILES_TO_RUN]
    
    print(f"Found {len(files_to_process)} file pairs to process.")

    # Loop through the first 10 files with a progress bar
    for img_path, lbl_path in tqdm(files_to_process, desc="Total Pipeline Progress"):
        
        # Create a unique output directory for this run
        # e.g., "PanTS_00000001"
        case_id = os.path.basename(os.path.dirname(img_path))
        output_dir = os.path.join("output", f"run_{case_id}")
        
        print(f"\n--- Processing Case: {case_id} ---")
        print(f"   -> Output Directory: {output_dir}")
        
        # Build the command as a list of strings
        command = [
            sys.executable, "central.py",
            "--input_image", img_path,
            "--input_label", lbl_path,
            "--output_dir", output_dir,
            "--epochs", str(EPOCHS),
            "--lr", str(LEARNING_RATE)
        ]
        
        # Run the command
        try:
            # We use subprocess.run here
            # check=True will raise an error if central.py fails
            subprocess.run(command, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {case_id}. The pipeline failed.")
            print(f"   Command was: {' '.join(command)}")
            print(f"   Return code: {e.returncode}")
            print("   Aborting the rest of the runs.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 User interrupted the process. Exiting.")
            sys.exit(1)

    print("\n✅ Successfully processed all {NUM_FILES_TO_RUN} files.")

if __name__ == "__main__":
    main()
