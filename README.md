# Visual Cochlea FFT Viewer (Anttis-FFT) (And image2spectra)

![fft](./anttisfft.png)

This is a simple Python script that provides a real-time, intuitive visualization of a webcam feed's frequency components.

It displays two panels side-by-side:

Position Space: The standard, raw webcam image (Y-Position vs. X-Position).

"Cochlea" Space: A hybrid visualization where the Y-axis is still position (image rows), but the X-axis shows the horizontal frequency components for that specific row.

# How It Works

Instead of a standard 2D Fast Fourier Transform (FFT) which transforms both X and Y axes into frequency, this script performs a 1D FFT on each row of the image independently (using np.fft.fft(gray, axis=1)).

This can be thought of as a "row-wise FFT" or a "hybrid space transform."

The result is a more intuitive map where you can see the frequency "ingredients" of the image (like sharp edges or repeating patterns) while still preserving their original vertical location.

What to Look For

Smooth/Blurry Surfaces: These are low-frequency. They will appear as a bright vertical line in the center of the Cochlea map (the 0-Hz line).

Sharp Vertical Edges: These are high-frequency. They will create bright "sparkles" or "wings" further out from the center on the Cochlea map, but still on the same row as the edge.

Repeating Horizontal Patterns: (Like window blinds) These will create distinct, bright spikes at a specific frequency (a specific x-position) on the map.

# Requirements

OpenCV (opencv-python)

NumPy (numpy)

PyQtGraph (pyqtgraph)

PyQt6 (PyQt6)

# How to Run

Ensure you have the required libraries installed:

pip install opencv-python numpy pyqtgraph PyQt6

# Run the script:

python anttis-fft.py
