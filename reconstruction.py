import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import imageio


# ---------- Reconstruction en 3D de la surface ----------
def reconstruct_3d_surface(registered_ilm, angles):

    angles_rad = np.radians(angles)
    w = len(registered_ilm[0])
    r_coords = np.arange(w)
    
    # Create a grid of points in polar coordinates
    theta_grid, r_grid = np.meshgrid(
        np.linspace(0, 2*np.pi, 100),  # 100 angles
        np.linspace(0, w//2, 100)      # 100 radii
    )
    
    # Calculate the depths at all available (r, theta) coordinates
    points = []
    values = []
    
    for i, ilm in enumerate(registered_ilm):
        angle = angles_rad[i]
        for r in range(w):
            points.append((angle, r))
            values.append(ilm[r])
    
    # Interpolate to create a continuous surface
    interp_points = np.column_stack((
        [p[0] for p in points],  # angles
        [p[1] for p in points]   # radii
    ))
    
    # Interpolate the surface
    grid_z = griddata(interp_points, values, (theta_grid, r_grid), method='nearest') # linear / nearest
    
    # Convert the grid to cartesian coordinates for visualization
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    
    surface_3d = {
        'x': x,
        'y': y,
        'z': grid_z
    }
    
    return surface_3d


def gaussian_surface(xy, amplitude, sigma_x, sigma_y, x0, y0, offset):
    x, y = xy
    x_centered = x - x0
    y_centered = y - y0
    
    # Gaussian function
    return amplitude * np.exp(-(x_centered**2/(2*sigma_x**2) + y_centered**2/(2*sigma_y**2))) + offset


def fit_mathematical_model(surface_3d):
    
    x = surface_3d['x'].flatten()
    y = surface_3d['y'].flatten()
    z = surface_3d['z'].flatten()
    
    valid = ~np.isnan(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    
    # Initialization
    p0 = [
        np.max(z) - np.min(z),  # amplitude
        20, 20,                  # sigma_x, sigma_y
        0, 0,                    # x0, y0
        np.min(z)                # offset
    ]
    
    # Fit the 2D Gaussian model
    params, _ = curve_fit(
        lambda xy, amp, sig_x, sig_y, x0, y0, offset: gaussian_surface((xy[0], xy[1]), amp, sig_x, sig_y, x0, y0, offset),
        (x, y), z, p0=p0, maxfev=10000
    )
    
    surface_model = "Gaussian Surface"
    model_params = {
        'amplitude': params[0],
        'sigma_x': params[1],
        'sigma_y': params[2],
        'x0': params[3],
        'y0': params[4],
        'offset': params[5]
    }
    
    print(f"\nParamètres de la gaussienne estimée : {model_params}\n")
        
    return surface_model, model_params


def plot_surface_graph(surface_3d, output_path, suffix):

    output_filename = f"fovea_surface{suffix}.png"
    output = os.path.join(output_path, output_filename)   
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        surface_3d['x'],
        surface_3d['y'],
        surface_3d['z'],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True
    )
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('Foveolar Depression 3D Surface')
    
    # Save the figure
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    return output


def plot_model_vs_actual(surface_3d, surface_model, model_params, output_path):
    
    output = os.path.join(output_path, "model_comparison.png")   
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    x = surface_3d['x']
    y = surface_3d['y']
    
    model_z = gaussian_surface(
        (x, y),
        model_params['amplitude'],
        model_params['sigma_x'],
        model_params['sigma_y'],
        model_params['x0'],
        model_params['y0'],
        model_params['offset']
    )
    
    
    fig = plt.figure(figsize=(15, 6))
    
    # Actual surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(x, y, surface_3d['z'], cmap=cm.viridis, alpha=0.8)
    ax1.set_title('Actual Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Depth')
    
    # Model surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(x, y, model_z, cmap=cm.plasma, alpha=0.8)
    ax2.set_title(f'Model: {surface_model}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Depth')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    return output


def save_model_parameters(surface_model, model_params, output_path, suffix):
    output_filename = f"model_parameters{suffix}.txt"
    output = os.path.join(output_path, output_filename) 
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    with open(output, 'w') as f:
        f.write(f"Mathematical Model: {surface_model}\n\n")
        f.write("Parameters:\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")
        
        # Add model equation
        f.write("\nModel Equation:\n")
        f.write("f(x, y) = amplitude * exp(-((x - x0)²/(2*sigma_x²) + (y - y0)²/(2*sigma_y²))) + offset\n")
    
    return output


def create_animated_gif(registered_images, output_path):

    output = os.path.join(output_path, "animated_oct.gif")   
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    frames = []
    
    for img in registered_images:
        frames.append((img * 255).astype(np.uint8))
    
    imageio.mimsave(output, frames, duration=1)
    
    return output