# Get all your training images
IMAGE_FILES=(PanTS/data/ImageTr/PanTS_*/ct.nii.gz)
LABEL_FILES=(PanTS/data/labels/PanTS_*/combined_labels.nii.gz)

# Loop through each one
for i in ${!IMAGE_FILES[@]}; do
    # Get the base ID, e.g., "PanTS_00000001"
    ID=$(basename $(dirname ${IMAGE_FILES[$i]}))
    
    echo "--- Processing $ID ---"
    
    # Define a unique output directory for this run
    OUTPUT_DIR="output/run_$ID"
    
    python central.py \
        --input_image "${IMAGE_FILES[$i]}" \
        --input_label "${LABEL_FILES[$i]}" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 300 \
        --lr 7e-4
done
