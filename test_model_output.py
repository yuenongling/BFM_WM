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


XLABEL=r'$u_1n_1/\nu$'
YLABEL=r'$u_pn_1/\nu$'
ZLABEL=r'$u_2n_1/\nu$'
OUTLABEL=r'$u_{\tau}n_1/\nu$'

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
surf = ax.plot_surface(x1_mesh.numpy(), x2_mesh.numpy(), output_xy_vary, cmap='viridis', edgecolor='none', rasterized=True)
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
    surf = ax.plot_surface(x1_mesh_xz.numpy(), x3_mesh_xz.numpy(), output_xz_vary, cmap='viridis', edgecolor='none', rasterized=True)
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
    ax_2d.contourf(x1_mesh_xz.numpy(), x3_mesh_xz.numpy(), output_xz_vary, levels=50, cmap='viridis', antialiased=True, linestyles='None', rasterized=True)
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
