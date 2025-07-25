'''
Test the output behavior of the model

1. 1D plot
2. 2D surface plot
3. Gradient summary
'''
import sys
import os
from src.wall_model import WallModel
from wall_model_cases import TURB_CASES, TURB_CASES_TREE, print_dataset_tree

sys.path.append("./paper_plots/")
from plot_utils import *

# Overwrite the default matplotlib settings
plt.rcParams.update({
    # "font.family": "serif",   # specify font family here
    # "font.serif": ["CMU Serif"],  # specify font here
    "font.size":21,
    # 'axes.labelsize': 'medium',
    # 'axes.titlesize':'medium',
    'text.usetex': True,  # Use LaTeX for text rendering
})          # specify font size here


XLABEL=r'$u_1y_1/\nu$'
YLABEL=r'$u_py_1/\nu$'
ZLABEL=r'$u_2y_1/\nu$'
OUTLABEL=r'$u_{\tau}y_1/\nu$'

model_path = sys.argv[1] if len(sys.argv) > 1 else None
IfSave = sys.argv[2] if len(sys.argv) > 2 else None

model_path = os.path.join('./models', model_path) if model_path else None
wall_model = WallModel.load_compact(model_path, device="cpu")
model = wall_model.model


# Test each dataset
print(f"\nDo a one-time loading of CH dataset to read in training data")
results = wall_model.test_external_dataset(
    dataset_key="CH",
    tauw=True,
    # mask_threshold=2e-4,
    save_path=None,
    purpose = 1,
    LogTransform=wall_model.config.get('training', {}).get('LogTransform', False),
)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # For 3D surface plots

# Adjust ranges based on desired input values
x_min, x_max = -500.0, 50000.0
y_min, y_max = -100.0, 400.0
z_min, z_max = -800.0, 50000.0
num_points_1d = 500
num_points_2d = 2000 # For 2D surface plots (can be computationally intensive)
num_samples_grad = 20000 # For gradient analysis

# --- Helper function for plotting style ---
def setup_plot(ax, title, xlabel, ylabel, zlabel=None):
    # ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=18, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=12)
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=18, labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # ax.grid(True)

# --- 1. 1D Slices ---
print("--- Plotting 1D Slices ---")
# Fix two variables, vary one
fixed_val_y, fixed_val_z = 0.0, 0.0
x_range = torch.linspace(x_min, x_max, num_points_1d)
inputs_x_vary = torch.stack([
    x_range,
    torch.full_like(x_range, fixed_val_y),
    torch.full_like(x_range, fixed_val_z)
], dim=1)

fixed_val_x, fixed_val_z_ = 0.0, 0.0 # Use z_ to avoid conflict
y_range = torch.linspace(y_min, y_max, num_points_1d)
inputs_y_vary = torch.stack([
    torch.full_like(y_range, fixed_val_x),
    y_range,
    torch.full_like(y_range, fixed_val_z_)
], dim=1)

fixed_val_x_, fixed_val_y_ = 0.0, 0.0 # Use x_, y_
z_range = torch.linspace(z_min, z_max, num_points_1d)
inputs_z_vary = torch.stack([
    torch.full_like(z_range, fixed_val_x_),
    torch.full_like(z_range, fixed_val_y_),
    z_range
], dim=1)

with torch.no_grad():
    output_x_vary = model(inputs_x_vary).squeeze().numpy()
    output_y_vary = model(inputs_y_vary).squeeze().numpy()
    output_z_vary = model(inputs_z_vary).squeeze().numpy()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].plot(x_range.numpy(), output_x_vary)
setup_plot(axs[0], f'Output vs Input 1 (Inputs 2,3 fixed at {fixed_val_y:.1f},{fixed_val_z:.1f})', XLABEL, OUTLABEL)
axs[1].plot(y_range.numpy(), output_y_vary)
setup_plot(axs[1], f'Output vs Input 2 (Inputs 1,3 fixed at {fixed_val_x:.1f},{fixed_val_z_:.1f})', YLABEL, OUTLABEL)
axs[2].plot(z_range.numpy(), output_z_vary)
setup_plot(axs[2], f'Output vs Input 3 (Inputs 1,2 fixed at {fixed_val_x_:.1f},{fixed_val_y_:.1f})', ZLABEL, OUTLABEL)
plt.tight_layout()
plt.show()

# --- 2. 2D Surface Plots ---
print("\n--- Plotting 2D Surface Plots (this might take a moment) ---")
# Fix one variable, vary two

# Vary x1, x2; fix x3
fixed_val_z_2d = 0.0
x1_mesh, x2_mesh = torch.meshgrid(
    torch.linspace(x_min, x_max, num_points_2d),
    torch.linspace(y_min, y_max, num_points_2d),
    indexing='ij'
)
inputs_xy_vary = torch.stack([
    x1_mesh.ravel(),
    x2_mesh.ravel(),
    torch.full_like(x1_mesh.ravel(), fixed_val_z_2d)
], dim=1)

with torch.no_grad():
    output_xy_vary = model(inputs_xy_vary).reshape(num_points_2d, num_points_2d).numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1_mesh.numpy(), x2_mesh.numpy(), output_xy_vary, cmap='viridis', edgecolor='none')
fig.colorbar(surf)
setup_plot(ax, f'Output vs Inputs 1,2 (Input 3 fixed at {fixed_val_z_2d:.1f})', XLABEL, YLABEL, OUTLABEL)
plt.show()

# --- 2.5 Plotting 2D Surface Plots with fixed y and varying x1, x3 ---

def plot_x_z_yfix(fixed_val_y_2d=0.0, save_fig=False):
        # You can repeat this for other pairs (vary x1, x3; fix x2 and vary x2, x3; fix x1)
# Example for varying x1, x3; fixing x2
    # fixed_val_y_2d = 0.0
    x1_mesh_xz, x3_mesh_xz = torch.meshgrid(
        torch.linspace(x_min, x_max, num_points_2d),
        torch.linspace(z_min, z_max, num_points_2d),
        indexing='ij'
    )
    inputs_xz_vary = torch.stack([
        x1_mesh_xz.ravel(),
        torch.full_like(x1_mesh_xz.ravel(), fixed_val_y_2d),
        x3_mesh_xz.ravel()
    ], dim=1)

    # breakpoint()

    with torch.no_grad():
        output_xz_vary = model(inputs_xz_vary).reshape(num_points_2d, num_points_2d).numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x1_mesh_xz.numpy(), x3_mesh_xz.numpy(), output_xz_vary, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, orientation='horizontal', location='top', fraction=0.05, label=OUTLABEL)
    setup_plot(ax, f'Output vs Inputs 1,3 (Input 2 fixed at {fixed_val_y_2d:.1f})', XLABEL, ZLABEL, OUTLABEL)
    z_level_for_scatter = 0
    # ax.scatter(
    #     wall_model.input[::10,0],  # Input 1 values
    #     wall_model.input[::10,2],  # Input 1 values
    #     zs=z_level_for_scatter, # Project onto this z-level
    #     zdir='z',             # Direction of projection is along z-axis
    #     c='red',              # Color of scatter points
    #     marker='.',           # Marker style
    #     s=15,                 # Marker size
    #     label='Data Points (projected)',
    #     alpha=0.6,             # Transparency
    #     rasterized=True # Use rasterized points for better performance on large datasets
    # )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0, 2000)
    ax.view_init(elev=25, azim=-65) # Top-down view

    fig_2d, ax_2d = plt.subplots(figsize=(8, 8))
    ax_2d.contourf(x1_mesh_xz.numpy(), x3_mesh_xz.numpy(), output_xz_vary, levels=50, cmap='viridis', antialiased=True, linestyles='None')
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(z_min, z_max)
    ax_2d.set_xlabel(XLABEL)
    ax_2d.set_ylabel(ZLABEL)

    if save_fig:
        fig.savefig(f'./paper_plots/plots/output_xz_vary_fixed_y_{fixed_val_y_2d}.pdf', bbox_inches='tight')
        fig_2d.savefig(f'paper_plots/plots/output_xz_vary_fixed_y_{fixed_val_y_2d}_2d.pdf', bbox_inches='tight')
    else:
        plt.show()

plot_x_z_yfix(0   , save_fig=IfSave)
plot_x_z_yfix(1   , save_fig=IfSave)
plot_x_z_yfix(10  , save_fig=IfSave)
plot_x_z_yfix(100, save_fig=IfSave)
plot_x_z_yfix(-100, save_fig=IfSave)

plt.close('all')

##
## # Example for varying x1, x3; fixing x2
## fixed_val_y_2d = 0.0
## x1_mesh_xz, x3_mesh_xz = torch.meshgrid(
##     torch.linspace(x_min, x_max, num_points_2d),
##     torch.linspace(z_min, z_max, num_points_2d),
##     indexing='ij'
## )
## inputs_xz_vary = torch.stack([
##     x1_mesh_xz.ravel(),
##     torch.full_like(x1_mesh_xz.ravel(), fixed_val_y_2d),
##     x3_mesh_xz.ravel()
## ], dim=1)
##
## with torch.no_grad():
##     output_xz_vary = model(inputs_xz_vary).reshape(num_points_2d, num_points_2d).numpy()
##
## fig = plt.figure(figsize=(8, 6))
## ax = fig.add_subplot(111, projection='3d')
## surf = ax.plot_surface(x1_mesh_xz.numpy(), x3_mesh_xz.numpy(), output_xz_vary, cmap='viridis', edgecolor='none')
## fig.colorbar(surf)
## setup_plot(ax, f'Output vs Inputs 1,3 (Input 2 fixed at {fixed_val_y_2d:.1f})', XLABEL, ZLABEL, OUTLABEL)
## plt.show()
##
#
## --- 3. Gradient Magnitude Analysis ---
#print("\n--- Analyzing Gradient Magnitudes ---")
## Sample points randomly in the ROI
#sample_inputs = torch.rand(num_samples_grad, 3) * torch.tensor([x_max-x_min, y_max-y_min, z_max-z_min]) + torch.tensor([x_min, y_min, z_min])
#sample_inputs.requires_grad_(True)
#
#outputs = model(sample_inputs)
#
## Compute gradients: dy/dx for each sample
#grad_outputs = torch.ones_like(outputs) # For scalar output
#gradients = torch.autograd.grad(outputs=outputs, inputs=sample_inputs, grad_outputs=grad_outputs, create_graph=False)[0]
## gradients will be a tensor of shape (num_samples_grad, 3)
#
#gradient_magnitudes = torch.norm(gradients, dim=1).detach().numpy()
#gradient_magnitudes_rela = torch.norm(gradients, dim=1).detach().numpy()/np.squeeze(outputs.detach().numpy())
#
#gradient_magnitudes_rela_input1 = torch.norm(gradients, dim=1).detach().numpy()/np.squeeze(sample_inputs[:,0].detach().numpy())
#
#plt.figure(figsize=(8, 6))
#plt.hist(gradient_magnitudes, bins=50, density=True, color='skyblue', edgecolor='black')
#plt.xlabel("Gradient Magnitude ||∇Output||")
#plt.ylabel("Density")
#plt.title(f"Distribution of Gradient Magnitudes (ReLU model)\nMean: {gradient_magnitudes.mean():.3f}, Max: {gradient_magnitudes.max():.3f}")
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.show()
#
#print(f"Mean gradient magnitude: {gradient_magnitudes.mean():.4f}")
#print(f"Median gradient magnitude: {np.median(gradient_magnitudes):.4f}")
#print(f"Max gradient magnitude: {gradient_magnitudes.max():.4f}")
#print(f"Min gradient magnitude: {gradient_magnitudes.min():.4f}")
#print(f"Std dev of gradient magnitude: {gradient_magnitudes.std():.4f}")
#
## Relative gradient to the corresponding output value
#plt.figure(figsize=(8, 6))
#plt.hist(gradient_magnitudes_rela, bins=50, density=True, color='skyblue', edgecolor='black')
#plt.xlabel("Relative Gradient Magnitude ||∇Output||")
#plt.ylabel("Density")
#plt.title(f"Distribution of Gradient Magnitudes (ReLU model)\nMean: {gradient_magnitudes.mean():.3f}, Max: {gradient_magnitudes.max():.3f}")
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.show()
#
#print(f"Relative Mean gradient magnitude: {gradient_magnitudes_rela.mean():.4f}")
#print(f"Relative Median gradient magnitude: {np.median(gradient_magnitudes_rela):.4f}")
#print(f"Relative Max gradient magnitude: {gradient_magnitudes_rela.max():.4f}")
#print(f"Relative Min gradient magnitude: {gradient_magnitudes_rela.min():.4f}")
#print(f"Relative Std dev of gradient magnitude: {gradient_magnitudes_rela.std():.4f}")
#
## --- 4. Hessian Analysis (Illustrative - for ReLU, mostly zero or undefined) ---
#print("\n--- Hessian Analysis (for a few points) ---")
#
## Use the x_min, x_max etc. defined earlier in your script for consistency
## If not defined, use some defaults e.g.
## x_min, x_max = -2.0, 2.0
## y_min, y_max = -2.0, 2.0
## z_min, z_max = -2.0, 2.0
#
#
#points_for_hessian = torch.tensor([
#    [0.0, 0.0, 0.0],
#    [0.5, 0.5, 0.5],
#    [1000,0,2000],
#    # Using a point within your previously defined ROI for consistency
#    # If x_min, y_min, z_min are not defined above, set them appropriately or use fixed values like [-1,-1,-1]
#    [globals().get('x_min', -1.0) / 2, globals().get('y_min', -1.0) / 2, globals().get('z_min', -1.0) / 2]
#], dtype=torch.float32)
#
#hessian_eigenvalues_all = []
#
## Check for torch.func.hessian (newer PyTorch versions, e.g., 2.0+)
#if hasattr(torch.func, 'hessian'):
#    print("Attempting to use torch.func.hessian")
#    # Import with an alias to be absolutely sure we're using the right one
#    from torch.func import hessian as torch_func_hessian_imported
#
#    # Define the function whose Hessian is to be computed.
#    # It must take a 1D tensor and return a scalar tensor.
#    def model_func_for_torch_func_hessian(x_single_1d_tensor):
#        # x_single_1d_tensor is expected to be a 1D tensor (e.g., shape (3,))
#        # wall_model expects a batch, so unsqueeze to (1, 3)
#        # .squeeze() ensures the output is a scalar tensor (0-dim)
#        return model(x_single_1d_tensor.unsqueeze(0)).squeeze()
#
#    try:
#        # Step 1: Create the Hessian calculator function (curried/transform style)
#        hessian_calculator = torch_func_hessian_imported(model_func_for_torch_func_hessian)
#        print(f"  Type of hessian_calculator (from torch.func): {type(hessian_calculator)}")
#
#        # Check if hessian_calculator is actually callable
#        if not callable(hessian_calculator):
#            print(f"  Error: hessian_calculator from torch.func.hessian is not callable. Type: {type(hessian_calculator)}")
#        else:
#            for i, point in enumerate(points_for_hessian):
#                print(f"\n  Calculating Hessian for point: {point.numpy()} using torch.func.hessian (curried call)")
#                try:
#                    # Step 2: Call the calculator with the specific point
#                    hess_matrix = hessian_calculator(point)
#                    print(f"    Type of hess_matrix after calling calculator: {type(hess_matrix)}")
#
#                    if not isinstance(hess_matrix, torch.Tensor):
#                        print(f"    Error: hess_matrix is not a torch.Tensor, but {type(hess_matrix)}. Skipping point.")
#                        continue
#
#                    # For scalar output func, Hessian should be (N,N).
#                    # If model_func_for_torch_func_hessian somehow returned (1,),
#                    # hess_matrix might be (1,N,N). Squeeze if necessary.
#                    if hess_matrix.ndim == 3 and hess_matrix.shape[0] == 1:
#                        hess_matrix = hess_matrix.squeeze(0)
#
#                    if hess_matrix.ndim != 2 or hess_matrix.shape[0] != point.shape[0] or hess_matrix.shape[1] != point.shape[0]:
#                        print(f"    Warning: Unexpected Hessian matrix shape: {hess_matrix.shape}. Expected ({point.shape[0]},{point.shape[0]})")
#
#
#                    print(f"    Hessian matrix at point {point.numpy()}:\n{hess_matrix.detach().numpy()}")
#                    # Use torch.linalg.eigvalsh for symmetric matrices (Hessian should be)
#                    eigenvalues = torch.linalg.eigvalsh(hess_matrix).detach().numpy()
#                    print(f"    Eigenvalues of Hessian: {eigenvalues}")
#                    hessian_eigenvalues_all.extend(eigenvalues)
#                except Exception as e_inner_loop:
#                    print(f"    Could not compute Hessian for point {point.numpy()} with torch.func.hessian (curried call): {e_inner_loop}")
#    except Exception as e_outer_calculator_creation:
#        print(f"  Could not create hessian_calculator with torch.func.hessian: {e_outer_calculator_creation}")
#        print(f"  This might happen if model_func_for_torch_func_hessian is problematic or PyTorch version has a different API for torch.func.hessian.")
#
#elif hasattr(torch.autograd.functional, 'hessian'): # For older PyTorch versions
#    print("Attempting to use torch.autograd.functional.hessian")
#    from torch.autograd.functional import hessian as torch_autograd_hessian_imported
#
#    def model_scalar_output_for_autograd_hessian(x_single_1d_tensor):
#        return model(x_single_1d_tensor.unsqueeze(0)).squeeze()
#
#    for i, point in enumerate(points_for_hessian):
#        print(f"\n  Calculating Hessian for point: {point.numpy()} using torch.autograd.functional.hessian")
#        try:
#            # This API should compute directly and return a tensor
#            hess_matrix = torch_autograd_hessian_imported(model_scalar_output_for_autograd_hessian, point)
#            print(f"    Type of hess_matrix (from torch.autograd.functional): {type(hess_matrix)}")
#
#            if not isinstance(hess_matrix, torch.Tensor):
#                print(f"    Error: hess_matrix is not a torch.Tensor, but {type(hess_matrix)}. Skipping point.")
#                continue
#
#            # No squeeze needed usually as this API is for scalar funcs giving (N,N) Hessian.
#            print(f"    Hessian matrix at point {point.numpy()}:\n{hess_matrix.detach().numpy()}")
#            eigenvalues = torch.linalg.eigvalsh(hess_matrix).detach().numpy()
#            print(f"    Eigenvalues of Hessian: {eigenvalues}")
#            hessian_eigenvalues_all.extend(eigenvalues)
#        except Exception as e_inner_loop_autograd:
#            print(f"    Could not compute Hessian for point {point.numpy()} with torch.autograd.functional: {e_inner_loop_autograd}")
#else:
#    print("Hessian computation skipped (torch.func.hessian or torch.autograd.functional.hessian not found).")
#    print("Consider updating PyTorch for this feature.")
#
## Plotting eigenvalues if any were collected
#if hessian_eigenvalues_all:
#    plt.figure(figsize=(8, 6))
#    plt.hist(hessian_eigenvalues_all, bins=20, density=True, color='lightcoral', edgecolor='black')
#    plt.xlabel("Hessian Eigenvalues")
#    plt.ylabel("Density")
#    plt.title("Distribution of Hessian Eigenvalues at Sampled Points (ReLU model)")
#    plt.grid(True, linestyle='--', alpha=0.7)
#    plt.show()
#    print("\nNote on Hessian Eigenvalues for ReLU models:")
#    print("- Theoretically, on flat linear segments, eigenvalues should be zero.")
#    print("- Near 'kinks' (where ReLUs change state), the Hessian is undefined. Numerical methods might yield varying results or errors at these exact points.")
#    print("- Non-zero eigenvalues indicate some local curvature. For a purely piecewise linear function, this would ideally be zero almost everywhere.")
#else:
#    print("\nNo Hessian eigenvalues were collected to plot.")
#
#print("\n--- Interpretation for ReLU models ---")
#print("1. 1D slices: Look for sharp 'corners' or 'kinks'. These are points where ReLUs switch state.")
#print("   The function between kinks will be linear.")
#print("2. 2D surfaces: Look for 'ridges' or 'valleys' formed by intersecting planar surfaces.")
#print("   The surface is made of many flat patches joined together.")
#print("3. Gradient Magnitudes: A wider distribution or higher maximum values suggest regions of rapid change.")
#print("   With ReLUs, you might see clusters of gradient magnitudes corresponding to the slopes of different linear pieces.")
#print("4. Hessian Eigenvalues: For perfect piecewise linear functions, eigenvalues should be zero on the flat pieces.")
#print("   Large eigenvalues would indicate high curvature (which ReLUs try to avoid locally, but can create globally).")
#print("   Numerical computations near kinks can sometimes produce non-zero values.")
#print("\nOverall: 'Smoothness' for ReLU networks means fewer, less sharp kinks and generally lower gradient magnitudes.")
#
## --- 5. 2D Surface Plot with Constraint: input3 > input1 ---
#print("\n--- Plotting 2D Surface Plot with Constraint: input3 > input1 ---")
#
## Define how much larger input3 should be than input1
## Adjust this delta based on your physical understanding.
## If input3 is typically much larger, delta could be larger.
## If it's just slightly larger, delta could be small.
#input3_delta_over_input1 = 1.1 # Example: input3 = input1 + 0.5
#constrained_x1_mesh, constrained_x2_mesh = torch.meshgrid(
#    torch.linspace(x_min, x_max, num_points_2d), # input1
#    torch.linspace(y_min, y_max, num_points_2d), # input2
#    indexing='ij'
#)
#
## Calculate input3 based on the constraint
#constrained_x3_values = constrained_x1_mesh * input3_delta_over_input1
#
#inputs_constrained_2d = torch.stack([
#    constrained_x1_mesh.ravel(),    # input1
#    constrained_x2_mesh.ravel(),    # input2
#    constrained_x3_values.ravel()   # input3 (derived from input1)
#], dim=1)
#
## with torch.no_grad():
##     output_constrained_2d = model(inputs_constrained_2d).reshape(num_points_2d, num_points_2d).numpy()
##
## fig_constrained = plt.figure(figsize=(10, 8)) # Slightly larger for better title visibility
## ax_constrained = fig_constrained.add_subplot(111, projection='3d')
## surf_constrained = ax_constrained.plot_surface(
##     constrained_x1_mesh.numpy(),
##     constrained_x2_mesh.numpy(),
##     output_constrained_2d,
##     cmap='viridis',
##     edgecolor='none',
##     zorder = 99,
##     alpha=0.8, # Slightly transparent for better visibility of scatter points
## )
## # Plot the scatter points on the "floor" (z = z_level_for_scatter)
## z_level_for_scatter = ax_constrained.get_zlim()[0] # Get current min z limit after plotting surface
## ax_constrained.scatter(
##     wall_model.input[::20,0],  # Input 1 values
##     wall_model.input[::20,1],  # Input 1 values
##     # wall_model.output[::20],  # Input 1 values
##     zs=z_level_for_scatter, # Project onto this z-level
##     zdir='z',             # Direction of projection is along z-axis
##     c='red',              # Color of scatter points
##     marker='.',           # Marker style
##     s=15,                 # Marker size
##     label='Data Points (projected)',
##     alpha=0.6,             # Transparency
##     rasterized=True # Use rasterized points for better performance on large datasets
## )
##
## fig_constrained.colorbar(surf_constrained, label='Output Value')
## ax_constrained.set_xlabel(r'$u_1y/\nu$')
## ax_constrained.set_ylabel(r'$u_py/\nu$')
## ax_constrained.set_zlabel(r'$u_{\tau}y/\nu$')
## ax_constrained.set_title(f'Output vs Inputs 1 & 2 (Constraint:  u2 = u1 * {input3_delta_over_input1})')
## ax_constrained.grid(True)
## ax_constrained.set_xlim(x_min, x_max)
## ax_constrained.set_ylim(y_min, y_max)
## ax_constrained.set_zlim(0, 800)
## ax.view_init(elev=25, azim=-65) # Top-down view
##     # plt.show()
## # Rotate the axes and update
## for angle in range(0, 360*4 + 1):
##     # Normalize the angle to the range [-180, 180] for display
##     angle_norm = (angle + 180) % 360 - 180
##
##     # Cycle through a full rotation of elevation, then azimuth, roll, and all
##     elev = azim = roll = 0
##     if angle <= 360:
##         elev = angle_norm
##     elif angle <= 360*2:
##         azim = angle_norm
##     elif angle <= 360*3:
##         roll = angle_norm
##     else:
##         elev = azim = roll = angle_norm
##
##     # Update the axis view and title
##     ax_constrained.view_init(elev, azim, roll)
##     plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
##
##     plt.draw()
##     plt.pause(.00001)
## # plt.show()
#
#print(f"\nPlotted constrained surface where input3 is set to input1 + {input3_delta_over_input1}.")
#print("Observations from this plot should be more relevant if this constraint holds in practice.")
#
## --- You can also do a 1D slice with this constraint ---
#print("\n--- Plotting 1D Slice with Constraint: input3 > input1 ---")
## Fix input2, vary input1, and set input3 based on input1
#fixed_val_input2_constrained = 0.0 # Example fixed value for input2
#input1_range_constrained = torch.linspace(x_min, x_max, num_points_1d)
#input3_values_constrained_1d = input1_range_constrained + input3_delta_over_input1
#
## Optional: clamp input3 if it goes out of expected range
## input3_values_constrained_1d = torch.clamp(input3_values_constrained_1d, min=z_min, max=z_max)
#
#
#inputs_constrained_1d_slice = torch.stack([
#    input1_range_constrained,
#    torch.full_like(input1_range_constrained, fixed_val_input2_constrained),
#    input3_values_constrained_1d
#], dim=1)
#
#with torch.no_grad():
#    output_constrained_1d_slice = model(inputs_constrained_1d_slice).squeeze().numpy()
#
#plt.figure(figsize=(8, 5))
#plt.plot(input1_range_constrained.numpy(), output_constrained_1d_slice)
#plt.xlabel("Input 1")
#plt.ylabel("Output")
#plt.title(f'Output vs Input 1 (Input 2 fixed at {fixed_val_input2_constrained}, Input 3 = Input 1 + {input3_delta_over_input1})')
#plt.grid(True)
#plt.show()
