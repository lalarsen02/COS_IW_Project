#!/usr/bin/env python

# ----------------------------------------------
# Research Assignment for my COS IW
# ----------------------------------------------

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.signal import stft
from scipy.io import wavfile

# -------------------------------------------------

def stereoToMono(data_stereo):
    # extract left and right channels
    left = data_stereo[:, 0]
    right = data_stereo[:, 1]

    # combine left and right channels to create mono signal
    mono = (left + right) / 2

    return mono

# First, turns wav forms into mel spectrograms
def create_spectrogram(wav_file_path):

    sample_rate, samples = wavfile.read(wav_file_path)
    if (len(samples.shape) == 2):
        samples = stereoToMono(samples)

    __, __, power_spectrum = stft(samples, sample_rate, 
                                               nperseg=1024, 
                                               noverlap=512, 
                                               nfft=1024*2, 
                                               window='hann', 
                                               detrend=False)

    mel_spec = librosa.feature.melspectrogram(S=power_spectrum, 
                                              sr=sample_rate, 
                                              n_mels=128, 
                                              fmin=0.0, 
                                              fmax=sample_rate / 2.0)
    
    mel_spec_db = librosa.power_to_db(np.abs(mel_spec)**2)

    return mel_spec_db, sample_rate

def create_spectrograms():
    training_X = []
    training_y = []
    testing_X = []
    testing_y = []

    base_path = 'Samples'

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # check if the file is a file (not a directory)
            if os.path.isfile(file_path):
                # call your function on the file
                mel_spec_db, sample_rate = create_spectrogram(file_path)

                plt.figure(figsize=(1, .4))
                librosa.display.specshow(mel_spec_db, cmap='inferno', sr=sample_rate, fmax=sample_rate / 2.0)

                rng = np.random.default_rng()
                rfloat = rng.random()

                if rfloat < 0.8:
                    file = file[:-4]
                    pic_path = os.path.join('Training', file)
                    plt.savefig(pic_path, bbox_inches='tight', pad_inches = 0)
                    pic_path = pic_path + '.png'
                    training_X.append(pic_path)
                    training_y.append(folder)

                else:
                    file = file[:-4]
                    pic_path = os.path.join('Testing', file)
                    plt.savefig(pic_path, bbox_inches='tight', pad_inches = 0)
                    pic_path = pic_path + '.png'
                    testing_X.append(pic_path)
                    testing_y.append(folder)
                
                plt.close()
    
    return training_X, training_y, testing_X, testing_y