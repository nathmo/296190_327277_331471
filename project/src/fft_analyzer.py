import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
# Base directory containing subfolders for each chocolate type.
base_dir = r'project\chocolate_data\praline_clean'  # Use a raw string to correctly handle backslashes

# Exclude these subfolders
exclude_folders = {"raw_praline", "references", "Background"}

# Define which component pairs to plot (using 1-indexed component numbers)
# For example: (1,2), (1,6), (1,10), (1,4), (2,4), (3,4)
pairs = [(1, 6), (3,7),(8,9)]

# -----------------------------
# Data Loading & FFT Processing
# -----------------------------
# List chocolate types from subfolders (excluding unwanted ones)
chocolate_types = sorted(
    [d for d in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, d)) and d not in exclude_folders]
)

# Prepare dictionaries to hold FFT amplitude and phase data for the first 16 modes.
# The structure is: data[mode_index][chocolate_type] = [list of values]
mode_amp = {mode: {choc: [] for choc in chocolate_types} for mode in range(16)}
mode_phase = {mode: {choc: [] for choc in chocolate_types} for mode in range(16)}

# Loop through each chocolate type subfolder
for choc in chocolate_types:
    subfolder = os.path.join(base_dir, choc)
    # List all PNG images (case-insensitive)
    image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(".png")]
    
    for img_file in image_files:
        img_path = os.path.join(subfolder, img_file)
        try:
            # Load image and convert to grayscale
            img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        img_array = np.array(img)
        
        # Compute the 2D FFT (complex values)
        fft_complex = np.fft.fft2(img_array)
        # Compute amplitude and phase
        amp = np.abs(fft_complex)
        phase = np.angle(fft_complex)
        # Option: Uncomment next lines to use log scaling for amplitude if needed.
        # amp = np.log1p(amp)
        
        # Extract the top-left 4x4 block, corresponding to the first 16 modes.
        for i in range(4):
            for j in range(4):
                mode_idx = i * 4 + j  # Flatten the 4x4 block into a single index (0 to 15)
                mode_amp[mode_idx][choc].append(amp[i, j])
                mode_phase[mode_idx][choc].append(phase[i, j])

# -----------------------------
# Modular Plotting Function (Amplitude & Phase)
# -----------------------------
def plot_fft_component_pairs_amp_phase(mode_amp, mode_phase, chocolate_types, pairs):
    """
    For each specified FFT component pair, create two scatter plots: one for amplitude and one for phase.
    
    Parameters:
      - mode_amp: Dictionary of amplitude values for each of the 16 modes per chocolate type.
      - mode_phase: Dictionary of phase values for each of the 16 modes per chocolate type.
      - chocolate_types: List of chocolate type names.
      - pairs: List of tuples (m1, m2) with 1-indexed component numbers.
    """
    # Create a colormap for distinct chocolate colors.
    cmap = plt.cm.get_cmap('tab20', len(chocolate_types))
    color_map = {choc: cmap(i) for i, choc in enumerate(chocolate_types)}
    
    n_pairs = len(pairs)
    # For each pair we will produce two plots (one amplitude, one phase),
    # so create a grid with 2 columns and n_pairs rows.
    ncols = 2
    nrows = n_pairs
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
    
    # Handle the case when there's only one pair (axes is not a 2D array)
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.reshape(nrows, ncols)
    
    # Loop over pairs: for each component pair, make two plots: amplitude and phase.
    for idx, (m1, m2) in enumerate(pairs):
        # Convert 1-indexed components to 0-indexed.
        comp1 = m1 - 1
        comp2 = m2 - 1
        
        # Get the corresponding axes for amplitude and phase.
        ax_amp = axes[idx, 0]
        ax_phase = axes[idx, 1]
        
        # Plot amplitude for this component pair.
        for choc in chocolate_types:
            x_vals_amp = mode_amp[comp1][choc]
            y_vals_amp = mode_amp[comp2][choc]
            ax_amp.scatter(x_vals_amp, y_vals_amp, color=color_map[choc], alpha=0.7, label=choc)
        ax_amp.set_title(f"Amplitude: Component {m1} vs Component {m2}")
        ax_amp.set_xlabel(f"Component {m1} Amplitude")
        ax_amp.set_ylabel(f"Component {m2} Amplitude")
        
        # Plot phase for this component pair.
        for choc in chocolate_types:
            x_vals_phase = mode_phase[comp1][choc]
            y_vals_phase = mode_phase[comp2][choc]
            ax_phase.scatter(x_vals_phase, y_vals_phase, color=color_map[choc], alpha=0.7, label=choc)
        ax_phase.set_title(f"Phase: Component {m1} vs Component {m2}")
        ax_phase.set_xlabel(f"Component {m1} Phase")
        ax_phase.set_ylabel(f"Component {m2} Phase")
    
    # Consolidate legends: create handles once.
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color_map[choc]) for choc in chocolate_types]
    labels = chocolate_types
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Call the plotting function with the desired pairs
# -----------------------------
plot_fft_component_pairs_amp_phase(mode_amp, mode_phase, chocolate_types, pairs)
