# Research Assignment for my COS IW
# First, turns wav forms into mel spectrograms
# -------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from playsound import playsound

# -------------------------------------------------

playsound("COS_IW_Project\Samples\Cymbals\CYCdh_Crash-01.wav")

# sample_rate, samples = wavfile.read(wav_file)
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
# print(times.shape, frequencies.shape, spectrogram.shape)

# plt.pcolormesh(times, frequencies, spectrogram, shading='flat', 
#                cmap='viridis', 
#                vmin=spectrogram.min(), vmax=spectrogram.max(), 
#                extent=(times[0], times[-1], frequencies[0], frequencies[-1]))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar(label='Power [dB]')
# plt.show()