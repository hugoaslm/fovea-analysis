import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


# ---------- Détection fovéa ----------
def detect_fovea_center(images, segmented_ilm, angles, save_dir):
    output = os.path.join(save_dir, "fovea_detected/")   
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fovea_centers = []
    
    h, w = images[0].shape
    search_w = int(w * 0.5)
    x_start = (w - search_w) // 2
    x_end = x_start + search_w

    for i, (img, ilm) in enumerate(zip(images, segmented_ilm)):
        # Restreindre la recherche dans une zone centrale
        cropped_ilm = ilm[x_start:x_end]
        rel_center_x = np.argmax(cropped_ilm)
        center_x = x_start + rel_center_x
        center_y = ilm[center_x]
        fovea_centers.append((center_x, center_y))

        # Annoter et sauvegarder l’image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img_rgb, (center_x, 0), (center_x, h - 1), (0, 0, 255), 2)
        cv2.putText(img_rgb, f"Fovea ({center_x}, {int(center_y)})", (center_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out_path = os.path.join(output, f"fovea_annotated_{i}.png")
        cv2.imwrite(out_path, img_rgb)

    return fovea_centers

# ---------- Recalage ----------
def register_images(images, segmented_ilm, segmented_hyp_hrc, segmented_ext_hrc, save_dir=None):
    print("Recalage images")
    num_images = len(images)
    H_ref, W_ref, _ = images[0].shape

    ref_img = images[0] / 255.0
    
    x_ilm = np.array(segmented_ilm[0])
    x_hyp_hrc = np.array(segmented_hyp_hrc[0])
    x_ext_hrc = np.array(segmented_ext_hrc[0])
    y = np.arange(len(x_ilm))

    registered_images = [ref_img]
    registered_ilm = [x_ilm]
    registered_hyp_hrc = [x_hyp_hrc]
    registered_ext_hrc = [x_ext_hrc]
    rmse_list = [0.0]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(1, num_images):
        img = images[i] / 255.0
        u_ilm = np.array(segmented_ilm[i])
        u_hyp_hrc = np.array(segmented_hyp_hrc[i])
        u_ext_hrc = np.array(segmented_ext_hrc[i])
        v = np.arange(len(u_ilm))

        # --- Première étape : recalage ILM + HRC ---
        x_all = np.concatenate([x_ilm, x_hyp_hrc, x_ext_hrc])
        u_all = np.concatenate([u_ilm, u_hyp_hrc, u_ext_hrc])
        y_all = np.concatenate([y, y, v])
        v_all = np.concatenate([v, v, v])

        N = len(x_all)
        A = np.zeros((N*2,6))
        B = np.zeros((N*2,1))
        
        for k in range(N):
            A[2*k, :] = np.array([x_all[k], y_all[k], 1, 0, 0, 0])
            A[2*k+1, :] = np.array([0, 0, 0, x_all[k], y_all[k], 1])
            B[2*k] = u_all[k]
            B[2*k+1] = v_all[k]
        
        h = np.linalg.inv(A.T @ A) @ A.T @ B
        
        H = np.array([[h[0][0], h[1][0], h[2][0]],
                      [h[3][0], h[4][0], h[5][0]],
                      [0, 0, 1]])
        
        P = np.array([[0,1,0],[1,0,0],[0,0,1]])
        tform = ski.transform.AffineTransform(P @ H @ P)
        registered_img = ski.transform.warp(img, tform, output_shape=(H_ref, W_ref))

        H_sym = P @ H @ P

        # Mise à jour des courbes recalées
        coords_ilm = np.stack([v, u_ilm, np.ones_like(v)])
        new_coords_ilm = H_sym @ coords_ilm
        aligned_ilm = new_coords_ilm[1, :]
        registered_ilm.append(aligned_ilm)
        
        coords_hyp_hrc = np.stack([v, u_hyp_hrc, np.ones_like(v)])
        new_coords_hyp_hrc = H_sym @ coords_hyp_hrc
        aligned_hyp_hrc = new_coords_hyp_hrc[1, :]
        registered_hyp_hrc.append(aligned_hyp_hrc)

        coords_ext_hrc = np.stack([v, u_ext_hrc, np.ones_like(v)])
        new_coords_ext_hrc = H_sym @ coords_ext_hrc
        aligned_ext_hrc = new_coords_ext_hrc[1, :]
        registered_ext_hrc.append(aligned_ext_hrc)

        # --- Deuxième étape : recalage image par cross-correlation ---
        shift_estimated, error, diffphase = ski.registration.phase_cross_correlation(ref_img, registered_img, upsample_factor=10)
        tform_shift = ski.transform.AffineTransform(translation=-shift_estimated[::-1])
        registered_img = ski.transform.warp(registered_img, tform_shift, output_shape=(H_ref, W_ref))

        registered_images.append(registered_img)

        dy, dx = shift_estimated[-2:]
        aligned_ilm -= dx
        aligned_hyp_hrc -= dx
        aligned_ext_hrc -= dx

        # Évaluation RMSE
        masque = registered_img > 0
        rmse = np.sqrt(np.sum((ref_img[masque] - registered_img[masque]) ** 2) / np.sum(masque))
        rmse_list.append(rmse)

        # Création de l'image RGB
        ref_img_gray = ski.color.rgb2gray(ref_img) 
        registered_img_gray = ski.color.rgb2gray(registered_img)
        
        im_rgb = np.zeros((H_ref, W_ref, 3), dtype=np.float32)
        im_rgb[..., 0] = ref_img_gray                         # red   channel
        im_rgb[..., 1] = registered_img_gray                  # green channel
        im_rgb[..., 2] = 0.5 * (ref_img_gray + registered_img_gray)  # blue  channel

        diff_img = np.abs(ref_img_gray - registered_img_gray)

        if save_dir:
            rgb_path  = os.path.join(save_dir, f"registration_{i}.png")
            plt.imsave(rgb_path,  (im_rgb * 255).astype(np.uint8))
        
        # Calcul de la moyenne et de l'écart-type de rmse_list
        mean_rmse = np.mean(rmse_list)
        stdev_rmse = np.std(rmse_list)
    
        if save_dir:
            stats_path = os.path.join(save_dir, "rmse_statistics.txt")
            with open(stats_path, 'w') as file:
                file.write(f"Mean RMSE: {mean_rmse}\n")
                file.write(f"Standard Deviation of RMSE: {stdev_rmse}\n")

    print("Valeurs de l'erreur quadratique moyenne par images :", rmse_list)
    return registered_images, rmse_list, registered_ilm, registered_hyp_hrc, registered_ext_hrc
