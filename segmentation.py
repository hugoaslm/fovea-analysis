import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from skimage import exposure, segmentation as ski_seg, transform as ski_tf, util as ski_util
import skimage as ski


# ---------- Prétraitement ----------
def load_images(image_dir):
    
    def extract_index(filename):
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else float('inf')
    
    print("Chargement des images...")
    
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.png')],
        key=extract_index
    )
    
    num_images = len(image_files)
    angle_step = 360 / num_images
    
    images = []
    angles = []
    
    for idx, file_name in enumerate(image_files):
        angle = idx * angle_step  # Compute angle based on sorted index
        full_path = os.path.join(image_dir, file_name)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        
        D = 496   # size of the eyefundus image
        H,W = img.shape
        B=50
        img = img[0:D,D-1:]
        img[-20:,-100:]=0       # mask text
        H,W = img.shape

        # Resize to same resolution
        if W < 1000:
            img = ski.transform.resize(img, [H, 2*W])
            img = ski.util.img_as_ubyte(img)
            H,W = img.shape
        
        images.append(img)
        angles.append(angle)
    
    sorted_data = sorted(zip(angles, images))
    angles, images = zip(*sorted_data)
    angles = list(angles)
    images = list(images)
    
    return images, angles


# ---------- Segmentation ----------
def customized_canny(img, low_thresh, high_thresh,
                     median_ksize,
                     clahe_clip=2.0,
                     clahe_grid=(8, 8),
                     bilateral_d=15,
                     bilateral_sigma_color=100,
                     bilateral_sigma_space=100):

    # 1) Médian
    med = cv2.medianBlur(img, median_ksize)

    # 2) CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    cl = clahe.apply(med)

    # 3) Bilateral pour réduire le bruit tout en gardant les bords
    blurred = cv2.bilateralFilter(cl,
                                  bilateral_d,
                                  bilateral_sigma_color,
                                  bilateral_sigma_space)

    # 4) Canny
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges

def vertical_gradient(img, ksize):
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_y = np.abs(grad_y)
    grad_y = (grad_y / grad_y.max()) * 255
    return grad_y.astype(np.uint8)

def build_cost_map(canny_edges, grad_map, alpha):
    canny_norm = canny_edges / 255.0
    grad_norm = grad_map / 255.0
    cost = 1.0 - (alpha * grad_norm + (1 - alpha) * canny_norm)
    return cost

def find_shortest_path(cost_map):
    h, w = cost_map.shape
    dp = np.zeros_like(cost_map)
    backtrack = np.zeros_like(cost_map, dtype=np.int32)
    path = np.zeros(w, dtype=np.int32)

    # Initialization
    dp[:, 0] = cost_map[:, 0]

    for col in range(1, w):
        mid = dp[:, col - 1]

        up = np.empty_like(mid)
        up[1:] = mid[:-1]
        up[0] = mid[0]

        down = np.empty_like(mid)
        down[:-1] = mid[1:]
        down[-1] = mid[-1]

        stack = np.stack([up, mid, down], axis=0)
        min_costs = np.min(stack, axis=0)
        argmins = np.argmin(stack, axis=0)

        # Fill dp and backtrack
        dp[:, col] = cost_map[:, col] + min_costs
        backtrack[:, col] = np.arange(h) + (argmins - 1)

    # Backtrack to find path
    path[-1] = np.argmin(dp[:, -1])
    for col in range(w - 2, -1, -1):
        path[col] = backtrack[path[col + 1], col + 1]

    return path

def smooth_path_polyfit(path, poly_order, center_proportion=0.9):
    n_points = len(path)
    x = np.arange(n_points)

    # Définir la fenêtre centrale
    start_index = int(n_points * (1 - center_proportion) / 2)
    end_index = int(n_points * (1 + center_proportion) / 2)

    # Si la fenêtre est vide, ou trop petite, prendre la totalité du chemin
    if end_index <= start_index:
        start_index = 0
        end_index = n_points

    x_center = x[start_index:end_index]
    path_center = path[start_index:end_index]

    # Ajustement polynomial sur la zone centrale
    coeffs = np.polyfit(x_center, path_center, poly_order)
    poly = np.poly1d(coeffs)
    smoothed = poly(x)

    return smoothed.astype(int)

def calibration(img, lower_percentile, upper_percentile):
    lower_bound = np.percentile(img, lower_percentile)
    upper_bound = np.percentile(img, upper_percentile)
    img_adjusted = ski.exposure.rescale_intensity(img, in_range=(lower_bound, upper_bound), out_range=(0, 255))
    return img_adjusted.astype(np.uint8)

def otsu_threshold(img):
    _, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_otsu

def active_contour(path, img):
    blurred = cv2.bilateralFilter(img, 15, 100, 100)
    H, W = img.shape
    init = np.zeros((2,W))
    init[1,:]= np.arange(W)
    init[0,:]= path
    init= init.T
    snake_ilm = ski.segmentation.active_contour(blurred, init, boundary_condition='fixed', alpha=2
                                                , w_edge=1, gamma=0.1)
    
    return snake_ilm


# ---------- Segmentation principale ----------
def segment_retinal_layers(images):
    segmented_ilm = []
    segmented_hyper_hrc = []
    segmented_ext_hrc = []
    
    print("Segmentation...")

    for img in images:

        #img = calibration(img, 4, 96)

        # ---- Segment ILM ----
        edges_ilm = customized_canny(img, 200, 255, 5)
        grad_map_ilm = vertical_gradient(img, ksize=29)
        cost_map_ilm = build_cost_map(edges_ilm, grad_map_ilm, 1)
        path_ilm = find_shortest_path(cost_map_ilm)

        window_size = 70
        std_devs = []
        for i in range(len(path_ilm) - window_size + 1):
            window = path_ilm[i:i + window_size]
            std_dev = np.std(window)
            std_devs.append(std_dev)
        jump_thresh = 10
            
        mean_intensity = img.mean()
        #print(mean_intensity)
        mean_thresh = 28
            
        if np.max(std_devs) > jump_thresh or mean_intensity < mean_thresh :
            img_otsu = otsu_threshold(img)
            edges_ilm = customized_canny(img_otsu, 200, 255, 5)
            grad_map_ilm = vertical_gradient(img_otsu, ksize=29)
            cost_map_ilm = build_cost_map(edges_ilm, grad_map_ilm, 1)
            path_ilm = find_shortest_path(cost_map_ilm)
            path_ilm = active_contour(path_ilm, img)
            path_ilm = path_ilm[:, 0].astype(np.uint8)
        

        # ---- First estimate of Hyper HRC for quality check ----
        edges_hyper_hrc = customized_canny(img, 80, 180, 7)
        grad_map_hyper_hrc = vertical_gradient(img, ksize=15)
        cost_map_hyper_hrc = build_cost_map(edges_hyper_hrc, grad_map_hyper_hrc, 0.8)

        cost_map_hrc_masked = cost_map_hyper_hrc.copy()
        for col in range(img.shape[1]):
            cost_map_hrc_masked[:path_ilm[col] + 10, col] = 1.0
        
        path_hyper_hrc = find_shortest_path(cost_map_hrc_masked)

        # ---- Segment Ext HRC ----
        edges_ext_hrc = customized_canny(img, 80, 180, 7)
        grad_map_ext_hrc = vertical_gradient(img, ksize=15)
        cost_map_ext_hrc = build_cost_map(edges_ext_hrc, grad_map_ext_hrc, 0.8)

        cost_map_ext_hrc_masked = cost_map_ext_hrc.copy()
        for col in range(img.shape[1]):
            cost_map_ext_hrc_masked[:path_hyper_hrc[col] + 10, col] = 1.0

        path_ext_hrc = find_shortest_path(cost_map_ext_hrc_masked)

        # --- Smoothing ---
        path_hyper_hrc = smooth_path_polyfit(path_hyper_hrc, poly_order=4)
        path_ext_hrc = smooth_path_polyfit(path_ext_hrc, poly_order=4)

        segmented_ilm.append(path_ilm)
        segmented_hyper_hrc.append(path_hyper_hrc)
        segmented_ext_hrc.append(path_ext_hrc)

    return segmented_ilm, segmented_hyper_hrc, segmented_ext_hrc


# ---------- Affichage ----------
def draw_and_save_all_segmentations(images, paths_ilm, paths_hyper_hrc, paths_ext_hrc, 
                                    output_dir):

    os.makedirs(output_dir, exist_ok=True)

    segmented_images = []

    for i, (img, ilm, hhrc, ehrc) in enumerate(zip(images, paths_ilm, paths_hyper_hrc, paths_ext_hrc)):
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape

        for col in range(w):
            if 0 <= ilm[col] < h:
                rgb_img[ilm[col], col] = [255, 0, 0]  # ILM - rouge
            if 0 <= hhrc[col] < h:
                rgb_img[hhrc[col], col] = [0, 255, 0]  # Hyper HRC - vert
            if 0 <= ehrc[col] < h:
                rgb_img[ehrc[col], col] = [0, 0, 255]  # Ext HRC - bleu

        segmented_images.append(rgb_img)

        filename = f"segmented_image_{i}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    return segmented_images

