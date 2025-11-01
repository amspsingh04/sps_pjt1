# evaluate_dice.py
import nibabel as nib
import numpy as np
import argparse
import sys

def calculate_dice(pred_mask, label_mask):
    """
    Calculates the Dice coefficient for a single class.
    """
    epsilon = 1e-6
    intersection = np.sum(pred_mask & label_mask)
    pred_sum = np.sum(pred_mask)
    label_sum = np.sum(label_mask)
    
    dice = (2.0 * intersection + epsilon) / (pred_sum + label_sum + epsilon)
    
    return dice

def main():
    parser = argparse.ArgumentParser(description="Calculate Dice score for multi-class NIfTI segmentations.")
    parser.add_argument("--pred", type=str, required=True, help="Path to your prediction .nii.gz file.")
    parser.add_argument("--label", type=str, required=True, help="Path to the ground-truth label .nii.gz file.")
    args = parser.parse_args()

    # --- 1. Load Files ---
    print(f"Loading Prediction: {args.pred}")
    try:
        pred_nii = nib.load(args.pred)
        pred_array = pred_nii.get_fdata().astype(np.int16)
    except FileNotFoundError:
        print(f"❌ FATAL: File not found at {args.pred}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Label: {args.label}")
    try:
        label_nii = nib.load(args.label)
        label_array = label_nii.get_fdata().astype(np.int16)
    except FileNotFoundError:
        print(f"❌ FATAL: File not found at {args.label}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Check Shapes ---
    if pred_array.shape != label_array.shape:
        print(f"❌ FATAL: Shape mismatch!")
        print(f"   -> Prediction Shape: {pred_array.shape}")
        print(f"   -> Label Shape:      {label_array.shape}")
        print("   -> Cannot compare. Did you forget to resample your prediction back to the original size?")
        sys.exit(1)
        
    print(f"   -> Shapes match: {pred_array.shape}")

    # --- 3. Calculate Multi-Class Dice ---
    unique_classes_label = np.unique(label_array)
    unique_classes_label = unique_classes_label[unique_classes_label != 0]

    unique_classes_pred = np.unique(pred_array)
    unique_classes_pred = unique_classes_pred[unique_classes_pred != 0]

    if len(unique_classes_label) == 0:
        print("⚠️ Warning: The ground-truth label file is empty (only contains background).")
        # If both are empty, that's a perfect score!
        if len(unique_classes_pred) == 0:
             print("   -> Prediction is also empty. Dice = 1.0")
             print("-" * 40)
             print(f"📊 Mean Dice (all classes): 1.0000")
             print("-" * 40)
        else:
             print("   -> Prediction found objects that aren't there. Dice = 0.0")
             print("-" * 40)
             print(f"📊 Mean Dice (all classes): 0.0000")
             print("-" * 40)
        sys.exit(0)
    
    # --- ❗ NEW FIX: Handle Empty Predictions ---
    if len(unique_classes_pred) == 0:
        print("❌ Prediction file is empty (all background).")
        print(f"   Ground truth has {len(unique_classes_label)} classes.")
        print("   Dice score for all classes is 0.")
        print("-" * 40)
        print(f"📊 Mean Dice (all classes): 0.0000")
        print("-" * 40)
        sys.exit(0)
    # --- END FIX ---

    print(f"\nFound {len(unique_classes_label)} classes to evaluate: {unique_classes_label}")
    
    dice_scores = []
    
    for class_id in unique_classes_label:
        pred_mask = (pred_array == class_id)
        label_mask = (label_array == class_id)
        
        dice_val = calculate_dice(pred_mask, label_mask)
        print(f"   -> Dice for Class {class_id}:  {dice_val:.4f}")
        dice_scores.append(dice_val)

    # --- 4. Print Final Results ---
    mean_dice = np.mean(dice_scores)
    
    print("-" * 40)
    print(f"📊 Mean Dice (all classes): {mean_dice:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
