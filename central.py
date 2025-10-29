# central.py

"""
A central orchestrator script to run the entire GNN segmentation pipeline
by calling each step as a subprocess.

Workflow:
1. Preprocessing (preprocessing.py)
2. Graph Building (graph_build.py)
3. Label Generation (generate_labels.py)
4. Model Training (train_unet.py)

Example usage from your terminal:
python central.py \
    --input_image "dataset/imagesTr/hippocampus_001.nii.gz" \
    --input_label "dataset/labelsTr/hippocampus_001.nii.gz" \
    --output_dir "output" \
    --epochs 300
"""

import subprocess
import sys
import argparse
import os

def run_preprocessing(input_nii, out_prefix):
    print("─" * 50)
    print("STEP 1/4: Running Preprocessing...")
    subprocess.run([
        sys.executable, "preprocessing.py",
        input_nii,
        "--out_prefix", out_prefix
    ], check=True)

def run_graph_build(volume_pkl, supervoxels_pkl, features_csv, out_graph):
    print("─" * 50)
    print("STEP 2/4: Running Graph Building...")
    subprocess.run([
        sys.executable, "graph_build.py",
        volume_pkl,
        supervoxels_pkl,
        features_csv,
        "--out_graph", out_graph
    ], check=True)

def run_label_generation(supervoxels_pkl, labels_nii, graph_path, out_map):
    print("─" * 50)
    print("STEP 3/4: Running Label Generation...")
    subprocess.run([
        sys.executable, "generate_labels.py",
        "--supervoxels_path", supervoxels_pkl,
        "--labels_nii_path", labels_nii,
        "--graph_path", graph_path,
        "--output_path", out_map
    ], check=True)

def run_training(graph_path, label_map_path, epochs, lr):
    print("─" * 50)
    print("STEP 4/4: Running Model Training...")
    subprocess.run([
        sys.executable, "train_unet.py",
        "--graph_path", graph_path,
        "--label_map_path", label_map_path,
        "--epochs", str(epochs),
        "--lr", str(lr)
    ], check=True)

def run_postprocessing(supervoxels_pkl, predictions_pt, node_map_pkl, original_nii, output_nii):
    """Calls the postprocess.py script."""
    print("─" * 50)
    print("STEP 5/5: Running Post-processing (Reprojection)...")
    subprocess.run([
        sys.executable, "postprocess.py",
        "--supervoxels_path", supervoxels_pkl,
        "--predictions_path", predictions_pt,
        "--node_map_path", node_map_pkl,
        "--original_image_nii", original_nii,
        "--output_nii", output_nii
    ], check=True)
    
def main():
    parser = argparse.ArgumentParser(description="Central orchestrator for the GNN pipeline.")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input raw NIfTI image file.")
    parser.add_argument('--input_label', type=str, required=True, help="Path to the input ground-truth NIfTI label file.")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save all generated files.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate for training.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preprocessed_prefix = os.path.join(args.output_dir, "preprocessed")
    volume_path = f"{preprocessed_prefix}_volume.pkl"
    supervoxels_path = f"{preprocessed_prefix}_supervoxels.pkl"
    features_path = f"{preprocessed_prefix}_features.csv"
    graph_path = os.path.join(args.output_dir, "supervoxel_graph.gpickle")
    label_map_path = os.path.join(args.output_dir, "supervoxel_label_mapping.pkl")
    node_map_path = os.path.join(args.output_dir, "node_mapping.pkl") # New path
    predictions_path = os.path.join(args.output_dir, "node_predictions.pt") # New path
    final_seg_path = os.path.join(args.output_dir, "final_segmentation.nii.gz")

    try:
        run_preprocessing(args.input_image, preprocessed_prefix)
        print("─" * 50)
        run_graph_build(volume_path, supervoxels_path, features_path, graph_path)
        print("─" * 50)
        run_label_generation(supervoxels_path, args.input_label, graph_path, label_map_path)
        print("─" * 50)
        run_training(graph_path, label_map_path, args.epochs, args.lr)
        print("─" * 50)
        run_postprocessing(supervoxels_path,predictions_path,node_map_path,args.input_image,final_seg_path)
        print("─" * 50)
        print("✅ Pipeline finished successfully!")
    except subprocess.CalledProcessError as e:
        print("─" * 50)
        print(f"❌ A step in the pipeline failed with exit code {e.returncode}.")
        print(f"   Command: {' '.join(e.cmd)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
