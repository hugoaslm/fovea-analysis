import os
import glob
import cv2
import numpy as np
import skimage as ski


def fill_between_layers(ilm_curve, ext_hrc_curve, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        y_top = int(ilm_curve[x]) if not np.isnan(ilm_curve[x]) else None
        y_bot = int(ext_hrc_curve[x]) if not np.isnan(ext_hrc_curve[x]) else None
        if y_top is not None and y_bot is not None and 0 <= y_top < y_bot < height:
            mask[y_top:y_bot+1, x] = 1
    return mask

def evaluate_segmentation(gt_dir, segmented_top, segmented_bot, name, output_file):
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    recall_list = []
    precision_list = []
    accuracy_list = []
    f1_list = []

    for i, gt_path in enumerate(gt_paths):
        # Chargement ground truth
        GT_bin = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        GT_bin = (GT_bin > 0).astype(np.uint8)
        height, width = GT_bin.shape
        
        # Resize to same resolution
        if width < 1000:
            GT_bin = ski.transform.resize(GT_bin, [height, 2*width])
            GT_bin = ski.util.img_as_ubyte(GT_bin)
            height, width = GT_bin.shape

        # Masque prédiction
        top_curve = segmented_top[i]
        bot_curve = segmented_bot[i]
        predicted_mask = fill_between_layers(top_curve, bot_curve, height, width)
        seg = (predicted_mask > 0).astype(np.uint8)

        GT = GT_bin

        # Calcul métriques
        TP = np.logical_and(GT == 1, seg == 1)
        TN = np.logical_and(GT == 0, seg == 0)
        FP = np.logical_and(GT == 0, seg == 1)
        FN = np.logical_and(GT == 1, seg == 0)

        tp_count = np.sum(TP)
        tn_count = np.sum(TN)
        fp_count = np.sum(FP)
        fn_count = np.sum(FN)

        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
        f1 = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0

        recall_list.append(recall)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        f1_list.append(f1)

        # Affichage image colorée
        color_seg = np.zeros((height, width, 3), dtype=np.uint8)
        color_seg[TP] = [255, 255, 255]
        color_seg[FP] = [255, 0, 0]
        color_seg[FN] = [0, 0, 255]

    # Résumé global
    print(f"\n=== Résultats globaux pour {name} (en %) ===")
    print(f"Recall    = {100*np.mean(recall_list):.2f} ± {100*np.std(recall_list):.2f}")
    print(f"Precision = {100*np.mean(precision_list):.2f} ± {100*np.std(precision_list):.2f}")
    print(f"Accuracy  = {100*np.mean(accuracy_list):.2f} ± {100*np.std(accuracy_list):.2f}")
    print(f"F1 Score  = {100*np.mean(f1_list):.2f} ± {100*np.std(f1_list):.2f}")

    with open(output_file, "a") as f:
        f.write(f"\n=== Résultats globaux pour {name} (en %) ===\n")
        f.write(f"Recall    = {100*np.mean(recall_list):.2f} ± {100*np.std(recall_list):.2f}\n")
        f.write(f"Precision = {100*np.mean(precision_list):.2f} ± {100*np.std(precision_list):.2f}\n")
        f.write(f"Accuracy  = {100*np.mean(accuracy_list):.2f} ± {100*np.std(accuracy_list):.2f}\n")
        f.write(f"F1 Score  = {100*np.mean(f1_list):.2f} ± {100*np.std(f1_list):.2f}\n")