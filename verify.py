import nibabel as nib
import numpy as np

# Make sure this is the correct path to your ground-truth label file
label_file_path = 'dataset/labelsTr/hippocampus_001.nii.gz'

try:
    label_img = nib.load(label_file_path)
    label_array = label_img.get_fdata()
    
    # Get the unique values and their counts
    unique_labels, counts = np.unique(label_array, return_counts=True)
    
    print(f"Found unique labels in '{label_file_path}':")
    for label, count in zip(unique_labels, counts):
        print(f"  - Label {int(label)}: {count} voxels")

except FileNotFoundError:
    print(f"Error: Could not find the file at '{label_file_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")