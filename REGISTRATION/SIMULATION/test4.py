import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d

def build_angular_signal(angle_list, resolution=360, blur_sigma=2.0):
    """
    Converts a list of angles (in degrees) into a circular signal vector.

    Parameters:
    - angle_list: list of angles in degrees
    - resolution: number of bins in the circular signal
    - blur_sigma: standard deviation for Gaussian blur (in bins)

    Returns:
    - signal: 1D numpy array representing the circular signal
    """
    signal = np.zeros(resolution)
    for angle in angle_list:
        idx = int(round(angle % 360)) % resolution
        signal[idx] += 1
    return gaussian_filter1d(signal, sigma=blur_sigma, mode='wrap')

def match_circular_signals(signal_a, signal_b):
    """
    Performs circular cross-correlation to find the best alignment.

    Parameters:
    - signal_a: 1D numpy array (reference)
    - signal_b: 1D numpy array (to be rotated)

    Returns:
    - best_offset: angle in degrees by which to rotate signal_b to align with signal_a
    - max_score: similarity score at best alignment
    - correlation: full cross-correlation array
    """
    correlation = correlate(signal_a, signal_b, mode='full', method='auto')
    n = len(signal_a)
    mid = len(correlation) // 2
    circular_corr = correlation[mid:mid+n]
    best_offset = np.argmax(circular_corr)
    max_score = circular_corr[best_offset]
    return best_offset, max_score, circular_corr

# Define angular ROI positions in degrees
angles_a = [10, 30, 70, 80, 290]  # cumulative from anchor
angles_b = [15, 25, 15]           # cumulative from anchor (wraps to 15)

# Convert to circular signals
signal_a = build_angular_signal(angles_a, resolution=360, blur_sigma=2)
signal_b = build_angular_signal(angles_b, resolution=360, blur_sigma=2)

# Match the signals
offset, score, correlation = match_circular_signals(signal_a, signal_b)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax[0].plot(signal_a, label="Plane A Signal")
ax[0].plot(np.roll(signal_b, offset), label=f"Plane B Signal (rotated by {offset}°)")
ax[0].set_title("Circular Angular Signals (Aligned)")
ax[0].legend()

ax[1].plot(correlation, label="Cross-Correlation")
ax[1].axvline(offset, color='r', linestyle='--', label=f"Best Offset: {offset}°")
ax[1].set_title("Cross-Correlation Profile")
ax[1].legend()

plt.tight_layout()
plt.show()

offset, score
