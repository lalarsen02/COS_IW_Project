# Research Assignment for my COS IW
# First, turns wav forms into mel spectrograms
# -------------------------------------------------

import matplotlib.pyplot as plt
import numpy
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
import pyaudio

# -------------------------------------------------

# from playsound import playsound

# playsound('CYCdh_Crash-01.wav')

sample_rate, samples = wavfile.read('Overhead Sample 1.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
print(times.shape, frequencies.shape, spectrogram.shape)

plt.pcolormesh(frequencies, times, numpy.transpose(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power [dB]')
plt.show()