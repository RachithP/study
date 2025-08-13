import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(0)

######################################################
    ################## 1D FFT  ##################
######################################################

# Generate a 1D signal (you can replace this with your data)
signal_length = 8
original_signal = np.sin(2 * np.pi * np.linspace(0, 4, signal_length))# + 0.5 * np.random.randn(signal_length)

# Perform 1D Discrete Fourier Transform
dft = np.fft.fft(original_signal)
frequencies = np.fft.fftfreq(signal_length)  # Frequency bins
magnitude = np.abs(dft)

# Sort by magnitude and retain the top 10 components
indices = np.argsort(-magnitude)  # Indices of components sorted by descending magnitude
top_indices = indices[:10]       # Top 10 components
filtered_dft = np.zeros_like(dft)  # Create an array of zeros for filtering
filtered_dft[top_indices] = dft[top_indices]  # Retain only top 10 components

# Perform Inverse Fourier Transform to reconstruct the signal
reconstructed_signal = np.fft.ifft(filtered_dft).real

# Visualize the original and reconstructed signals as images
plt.figure(figsize=(15, 10))

# Original signal
plt.subplot(3, 1, 1)
plt.title("Original Signal (Visualized as Image)")
# plt.imshow(original_signal.reshape(1, -1), cmap='viridis', aspect='auto')
# plt.colorbar(label="Amplitude")
# plt.axis('off')
plt.stem( 2 * np.pi * np.linspace(0, 4, signal_length), original_signal)
plt.grid(True)

# Top Fourier components
plt.subplot(3, 1, 2)
plt.title("Top 10 Fourier Components (Magnitudes)")
plt.stem(frequencies[top_indices], magnitude[top_indices])
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

# Reconstructed signal
plt.subplot(3, 1, 3)
plt.title("Reconstructed Signal with Top 10 Components (Visualized as Image)")
# plt.imshow(reconstructed_signal.reshape(1, -1), cmap='viridis', aspect='auto')
# plt.colorbar(label="Amplitude")
# plt.axis('off')
plt.stem(2 * np.pi * np.linspace(0, 4, signal_length), reconstructed_signal)
plt.grid(True)

plt.tight_layout()
plt.show()


######################################################
    ################## 2D FFT  ##################
######################################################

# Load the image in grayscale
# image = cv2.imread('D:/github/tifa/image_generation/sd2/coco_301091.png', cv2.IMREAD_GRAYSCALE)

# # Perform the Fourier Transform
# f_transform = np.fft.fft2(image)  # 2D Fourier Transform
# f_shift = np.fft.fftshift(f_transform)  # Shift the zero-frequency component to the center
# magnitude_spectrum = 20 * np.log(np.abs(f_shift))  # Compute the magnitude spectrum

# # Display the original image and its Fourier Transform
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Magnitude Spectrum")
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.axis('off')

# plt.tight_layout()
# plt.show()