#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:57:54 2024
@author: wanglukai
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_system_matrix(grid_size, num_strips=32, num_rotations=16):
    """
    Generate the system matrix for a given grid size.
    """
    dy_strip = 2 / num_strips  # Strip width
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    mask = (X**2 + Y**2) <= 1  # Disk mask
    pixel_area = (2 / grid_size)**2  # Area of each pixel

    num_sensitivity_functions = num_strips * num_rotations
    H = np.zeros((num_sensitivity_functions, grid_size**2))

    for k in range(num_rotations):
        theta = k * np.pi / num_rotations
        for m in range(num_strips):
            y_start = -1 + m * dy_strip
            y_end = y_start + dy_strip
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
            sensitivity_mask = (Y_rot >= y_start) & (Y_rot < y_end) & mask
            row_index = k * num_strips + m
            H[row_index, :] = sensitivity_mask.flatten() * pixel_area

    return H


def compute_pseudo_inverse(H, tol=1e-10):
    """
    Compute the pseudo-inverse of the system matrix H using SVD.
    """
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    S_inv = np.zeros_like(S)
    S_inv[S > tol] = 1 / S[S > tol]
    H_pseudo_inv = Vt.T @ np.diag(S_inv) @ U.T
    return H_pseudo_inv


def downsample_object(obj, target_size):
    """
    Downsample a 2D object to the target size.
    """
    original_size = int(np.sqrt(obj.size))  # Infer original size
    obj_2d = obj.reshape(original_size, original_size)  # Reshape to 2D
    obj_downsampled = resize(obj_2d, (target_size, target_size), mode='reflect', anti_aliasing=True)
    return obj_downsampled.flatten()


def save_array_to_folder(array, folder, prefix, idx):
    """
    Save a single array to a file within a folder.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{prefix}_{idx}.npy")
    np.save(file_path, array)


def visualize_random_images(grid_size, original_objects, reconstructions, num_to_show=10):
    """
    Randomly select and visualize original and reconstructed images.
    """
    indices = np.random.choice(len(original_objects), num_to_show, replace=False)
    plt.figure(figsize=(20, 8))

    for i, idx in enumerate(indices):
        original = original_objects[idx].reshape(grid_size, grid_size)
        reconstructed = reconstructions[idx].reshape(grid_size, grid_size)

        # Original image
        plt.subplot(2, num_to_show, i + 1)
        plt.imshow(original, extent=(-1, 1, -1, 1), cmap="hot")
        plt.title(f"Original {idx}")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(2, num_to_show, i + 1 + num_to_show)
        plt.imshow(reconstructed, extent=(-1, 1, -1, 1), cmap="hot")
        plt.title(f"Reconstructed {idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def process_grid_size(grid_size, object_dir="object_realizations", output_dir="proj_rec", num_to_show=5):
    """
    Process a specific grid size: Generate system matrix, compute pseudo-inverse,
    process objects, and visualize results.
    """
    print(f"Processing grid size: {grid_size}x{grid_size}")

    # Generate and save the system matrix
    H = generate_system_matrix(grid_size)
    np.save(f"H_{grid_size}.npy", H)
    print(f"System matrix saved as H_{grid_size}.npy")

    # Compute and save the pseudo-inverse
    H_pseudo_inv = compute_pseudo_inverse(H)
    np.save(f"H_pseudo_inv_{grid_size}.npy", H_pseudo_inv)
    print(f"Pseudo-inverse saved as H_pseudo_inv_{grid_size}.npy")

    # Load and process objects
    object_files = sorted(
        [f for f in os.listdir(object_dir) if f.startswith("object_") and f.endswith(".npy")]
    )

    original_objects = []
    projections = []
    reconstructions = []

    for idx, obj_file in enumerate(object_files):
        obj = np.load(os.path.join(object_dir, obj_file))
        obj_downsampled = downsample_object(obj, grid_size)

        proj = H @ obj_downsampled  # Compute projection data
        rec = H_pseudo_inv @ proj  # Reconstruct the object

        # Save projections and reconstructions by index
        save_array_to_folder(proj, output_dir, f"projection_{grid_size}", idx)
        save_array_to_folder(rec, output_dir, f"reconstruction_{grid_size}", idx)

        original_objects.append(obj_downsampled)
        projections.append(proj)
        reconstructions.append(rec)

    print(f"Projections and reconstructions saved in '{output_dir}' for grid size {grid_size}x{grid_size}")

    # Visualize random samples
    visualize_random_images(grid_size, original_objects, reconstructions, num_to_show=num_to_show)


if __name__ == "__main__":
    grid_sizes = [64, 32, 8]
    for grid_size in grid_sizes:
        process_grid_size(grid_size)

    print("All processing completed.")
