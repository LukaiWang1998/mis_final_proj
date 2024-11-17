import os
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(grid_size=128, A_s=3):
    """
    Generate the signal component.
    """
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # Sample signal parameters
    c_xs, c_ys = np.random.uniform(-0.5, 0.5, 2)  # Center of the signal
    sigma_s = np.random.uniform(0, 0.5)           # Radius of the signal

    # Create the circular signal
    distance = np.sqrt((X - c_xs)**2 + (Y - c_ys)**2)
    circ = distance <= sigma_s
    f_s = A_s * circ

    return f_s, circ, sigma_s  # Return the signal, mask, and radius for labels

def generate_background(grid_size=128, N=10, sigma_s=None):
    """
    Generate the background component.
    """
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    f_b = np.zeros_like(X)

    for _ in range(N):
        c_xn, c_yn = np.random.uniform(-0.5, 0.5, 2)  # Lump center
        gaussian = np.exp(-((X - c_xn)**2 + (Y - c_yn)**2) / (2 * sigma_s**2)) / (2 * np.pi * sigma_s**2)
        f_b += gaussian

    return f_b

def generate_object(grid_size=128, N=10):
    """
    Generate the complete object with signal and background.
    """
    f_s, signal_mask, sigma_s = generate_signal(grid_size)
    f_b = generate_background(grid_size, N, sigma_s)
    f = f_s + f_b
    return f, signal_mask

def generate_segmentation_labels(signal_mask):
    """
    Generate segmentation labels: 1 for background, 2 for signal.
    """
    labels = np.ones_like(signal_mask, dtype=int)
    labels[signal_mask] = 2
    return labels

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Directory for saving results
output_dir = "object_realizations"
os.makedirs(output_dir, exist_ok=True)

# Parameters
grid_size = 128
N = 10
J = 500  # Number of realizations

for j in range(J):
    # Generate object and segmentation labels
    f, signal_mask = generate_object(grid_size, N)
    labels = generate_segmentation_labels(signal_mask)

    # Save the object and labels
    np.save(os.path.join(output_dir, f"object_{j}.npy"), f)
    np.save(os.path.join(output_dir, f"labels_{j}.npy"), labels)

    # Optionally print progress
    if (j + 1) % 50 == 0:
        print(f"Saved {j + 1}/{J} realizations.")

print(f"All {J} realizations have been saved to '{output_dir}'.")

# Randomly showcase 10 cases

# List all saved objects and labels
object_files = [f for f in os.listdir(output_dir) if f.startswith("object_") and f.endswith(".npy")]
label_files = [f for f in os.listdir(output_dir) if f.startswith("labels_") and f.endswith(".npy")]

# Ensure the files are sorted (so objects and labels match by index)
object_files.sort()
label_files.sort()

# Randomly sample 10 indices
np.random.seed(42)  # Set seed for reproducibility
sampled_indices = np.random.choice(len(object_files), 10, replace=False)

# Visualize the sampled objects and labels
plt.figure(figsize=(20, 8))

for i, idx in enumerate(sampled_indices):
    # Load object and label
    obj_path = os.path.join(output_dir, object_files[idx])
    lbl_path = os.path.join(output_dir, label_files[idx])
    obj = np.load(obj_path)
    lbl = np.load(lbl_path)

    # Plot the object
    plt.subplot(2, 10, i + 1)
    plt.imshow(obj, extent=(-1, 1, -1, 1), cmap="hot")
    plt.title(f"Object {idx}")
    plt.axis("off")

    # Plot the segmentation labels
    plt.subplot(2, 10, i + 11)
    plt.imshow(lbl, extent=(-1, 1, -1, 1), cmap="viridis")
    plt.title(f"Labels {idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()


