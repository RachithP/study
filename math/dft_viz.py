import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal: A sum of two sine waves
t = np.linspace(0, 1, 500, endpoint=False)  # Time variable
freq1, freq2 = 5, 20  # Frequencies in Hz
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Perform Fourier Transform
dft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=t[1] - t[0])
magnitude = np.abs(dft)

# Plot Original Signal
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
plt.title("Original Signal (Time Domain)")
plt.plot(t, signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot Frequency Spectrum (Magnitude)
plt.subplot(3, 1, 2)
plt.title("Magnitude Spectrum (Frequency Domain)")
plt.stem(frequencies[:len(frequencies)//2], magnitude[:len(frequencies)//2])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

# Reconstruct Signal from Top Components
top_components = 3  # Retain top n frequencies, the more you retain the better reconstruction
sorted_indices = np.argsort(-magnitude)
filtered_dft = np.zeros_like(dft)
filtered_dft[sorted_indices[:top_components]] = dft[sorted_indices[:top_components]]
reconstructed_signal = np.fft.ifft(filtered_dft).real

# Plot Reconstructed Signal
plt.subplot(3, 1, 3)
plt.title("Reconstructed Signal with Top Components")
plt.plot(t, reconstructed_signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
