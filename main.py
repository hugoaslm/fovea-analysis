import os
from segmentation import (
    load_images,
    segment_retinal_layers,
    draw_and_save_all_segmentations,
)
from recalage import detect_fovea_center, register_images
from reconstruction import (
    reconstruct_3d_surface,
    fit_mathematical_model,
    plot_surface_graph,
    plot_model_vs_actual,
    create_animated_gif,
    save_model_parameters,
)
from evaluation import evaluate_segmentation

# Séries : # 01_CAB_OD, 02_DEA_OD, 03_DEA_OS, 04_BIM_OD, 05_MIG_OD, 06_MIG_OS, 07_TIM_OD, 08_TRK_OD
subfolders = [
    "03_DEA_OS"
]

for subfolder in subfolders:
    print(f"\n--- Processing {subfolder} ---")

    # Bien mettre le fichier des images dans le même dossier que le projet !
    image_dir = os.path.join("IMAGES", subfolder)
    
    output_dir = os.path.join("output/median_ksize", subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load images
    images, angles = load_images(image_dir)

    # Step 2: Segment retinal layers
    segmented_ilm, segmented_hyper_hrc, segmented_ext_hrc = segment_retinal_layers(images)

    # Step 2 bis: Draw the segmentation
    segmented_images = draw_and_save_all_segmentations(
        images=images,
        paths_ilm=segmented_ilm,
        paths_hyper_hrc=segmented_hyper_hrc,
        paths_ext_hrc=segmented_ext_hrc,
        output_dir=os.path.join(output_dir, "segmented_images")
    )

    # Step 3: Detect fovea center
    fovea_center = detect_fovea_center(images, segmented_ilm, angles, output_dir)
    
    # Step 4: Register images
    registered_images, rmse_list, registered_ilm, registered_hyp_hrc, registered_ext_hrc  = register_images(
        segmented_images,
        segmented_ilm,
        segmented_hyper_hrc,
        segmented_ext_hrc,
        save_dir=os.path.join(output_dir, "registered_images")
    )
    
    # Préparation pour le traitement des trois couches
    layer_names = ["ilm", "hyp_hrc", "ext_hrc"]
    segmented_layers = [segmented_ilm, segmented_hyper_hrc, segmented_ext_hrc]
    registered_layers = [registered_ilm, registered_hyp_hrc, registered_ext_hrc]
    surface_3d = {}
    surface_models = {}
    model_params_all = {}
    surface_graph_paths = []
    params_paths = []
    
    for name, registered in zip(layer_names, registered_layers):
        # Step 5: Reconstruct 3D surface
        surface = reconstruct_3d_surface(registered, angles)
        surface_3d[name] = surface
    
        # Step 6: Fit mathematical model
        model, params = fit_mathematical_model(surface)
        surface_models[name] = model
        model_params_all[name] = params
    
        # Step 7: Plot surface graph
        surface_path = plot_surface_graph(surface, output_dir, suffix=f"_{name}")
        surface_graph_paths.append(surface_path)
    
        # Save model parameters
        param_path = save_model_parameters(model, params, output_dir, suffix=f"_{name}")
        params_paths.append(param_path)
    
    # Générer le GIF
    gif_path = create_animated_gif(registered_images, output_dir)
    
    # Comparaison modèle vs surface ILM
    model_path = plot_model_vs_actual(
        surface_3d["ilm"],
        surface_models["ilm"],
        model_params_all["ilm"],
        output_dir
    )
    
    # Collecte des résultats
    results = {
        "animated_gif": gif_path,
        "surface_graphs": surface_graph_paths,
        "model_comparison": model_path,
        "model_parameters": params_paths
    }
    
    # Évaluation
    output_file = os.path.join(output_dir, "segmentation_metrics.txt")
    
    with open(output_file, "w") as f:
        f.write("Résultats des évaluations de segmentation\n")
    
    # Évaluation ILM → EXT_HRC (R_BIN1)
    evaluate_segmentation(
        gt_dir=os.path.join("IMAGES", "R_BIN1", subfolder),
        segmented_top=segmented_ilm,
        segmented_bot=segmented_ext_hrc,
        name="ILM to EXT_HRC",
        output_file=output_file
    )
    
    # Évaluation HYPER_HRC → EXT_HRC (R_BIN2)
    evaluate_segmentation(
        gt_dir=os.path.join("IMAGES", "R_BIN2", subfolder),
        segmented_top=segmented_hyper_hrc,
        segmented_bot=segmented_ext_hrc,
        name="HYPER_HRC to EXT_HRC",
        output_file=output_file
    )
    
    print(f"\nTraitement terminé pour la série : {subfolder}!")
    print("Fichiers de sortie :")
    for output_type, paths in results.items():
        if isinstance(paths, list):
            for path in paths:
                print(f"- {output_type}: {path}")
        else:
            print(f"- {output_type}: {paths}")

