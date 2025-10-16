import subprocess
import sys

def run_preprocessing(input_nii, out_prefix):
    print("Running preprocessing...")
    subprocess.run([
        sys.executable, "preprocessing.py",
        input_nii,
        "--out_prefix", out_prefix
    ], check=True)

def run_graph_build(volume_pkl, supervoxels_pkl, features_csv, out_graph):
    print("Running graph building...")
    subprocess.run([
        sys.executable, "graph_build.py",
        volume_pkl,
        supervoxels_pkl,
        features_csv,
        "--out_graph", out_graph
    ], check=True)


def main():
    input_nii = "PanTS/data/ImageTr/PanTS_00000001/ct.nii.gz" #only thing that needs to change
    out_prefix = "output/preprocessed"
    graph_out = "output/supervoxel_graph.gpickle"

    run_preprocessing(input_nii, out_prefix)
    run_graph_build(f"{out_prefix}_volume.pkl", f"{out_prefix}_supervoxels.pkl", f"{out_prefix}_features.csv", graph_out)

if __name__ == "__main__":
    main()

