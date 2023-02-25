import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import wave
from IPython.display import Audio

def visualize_audio(path):
    # Load audio signal
    with wave.open(path, "rb") as wave_file:
        signal = wave_file.readframes(wave_file.getnframes())
        signal = np.frombuffer(signal, dtype=np.int16)

    # Plot the time-domain signal
    plt.figure(figsize=(15, 4))
    plt.plot(signal, color='blue')
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.title("Time-domain Signal "+path.split('/')[-2])

    # Apply FFT to the audio signal
    fft_signal = np.abs(scipy.fftpack.fft(signal))

    # Plot the frequency-domain signal
    plt.figure(figsize=(15, 4))
    plt.plot(fft_signal, color='red')
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.title("Frequency-domain Signal "+path.split('/')[-2])

    # Compute spectrogram
    plt.figure(figsize=(15, 4))
    NFFT = 1024
    Fs = 44100
    noverlap = NFFT // 2
    _, _, _, spec = plt.specgram(signal, NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap='viridis')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram "+path.split('/')[-2])
    plt.colorbar(format="%+2.0f dB")

    plt.show()

def play_audio(path):
    display(Audio(path))
