import numpy as np

def generate_discrete_grid(grid_size=128):
    """
    Generate a 2D grid of coordinates.
    """
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def generate_signal(grid_size=128, A_s=3):
    """
    Generate the signal component.
    """
    # Sample signal parameters
    c_xs, c_ys = np.random.uniform(-0.5, 0.5, 2)  # Center of the signal
    sigma_s = np.random.uniform(0, 0.5)           # Radius of the signal

    # Discretize over grid pixels
    X, Y = generate_discrete_grid(grid_size)

    # Create the circular signal
    distance = np.sqrt((X - c_xs)**2 + (Y - c_ys)**2)
    circ = distance <= sigma_s
    f_s = A_s * circ

    return f_s, circ, sigma_s  # Return the signal, mask, and radius for labels

def generate_background(grid_size=128, N=10, sigma_s=None):
    """
    Generate the background component.
    """
    X, Y = generate_discrete_grid(grid_size)

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
