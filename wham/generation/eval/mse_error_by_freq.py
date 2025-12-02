import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import defaultdict

from wam import CODA_DIR, REGEN_CODA_DIR

TARGET_SAMPLERATE = 44100

"""
This script takes two sets of paired .wav files and parses each file, finding the mean error between the fourier
transforms of each file. It then creates a graph of the mean error by frequency.
"""

# Compute FFT from raw data and samplerate
def compute_fft_from_data(data, samplerate=TARGET_SAMPLERATE, window_size=1500, hop_size=1000):
    if len(data.shape) > 1:
        data = data[:, 0]  # Convert to mono if stereo

    # Window function
    window = np.hanning(window_size)
    
    # Number of frames
    num_frames = 1 + (len(data) - window_size) // hop_size
    spectrogram = []

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = data[start:end]
        
        if len(frame) < window_size:
            break  # Ignore incomplete window at end
        
        frame = frame * window
        fft_frame = np.fft.fft(frame)
        magnitude = np.abs(fft_frame[:window_size // 2])  # Keep only positive freqs
        spectrogram.append(magnitude)

    spectrogram = np.array(spectrogram)
    avg_spectrum = np.mean(spectrogram, axis=0)  # Average over time

    freqs = np.fft.fftfreq(window_size, d=1/samplerate)[:window_size // 2]
    return freqs, avg_spectrum


def calculate_mse(freq1, freq2, fft1, fft2):
    min_freq = max(freq1[0], freq2[0])
    max_freq = min(freq1[-1], freq2[-1])

    common_freqs = np.linspace(min_freq, max_freq, num=1000)  # fixed number of points for comparison

    fft1_interp = np.interp(common_freqs, freq1, fft1)
    fft2_interp = np.interp(common_freqs, freq2, fft2)

    mse_per_freq = (fft1_interp - fft2_interp) ** 2 /(fft1_interp**2) # element-wise squared error
    common_freqs = np.round(common_freqs / 10) * 10
    return common_freqs, mse_per_freq

def process_files(original_dir, gen_dir):
    """Calculate average mse error between both sets of audio files. This script parses through every file in 
    original_dir, and finds the corresponding file in gen_dir. The code expects each file in original_dir to be in 
    the form file_name.wav, and each corresponding file in gen_dir to be in the form file_name_gen.dir
    """
    mse_values_by_freq = []

    for filename in os.listdir(original_dir):
        if filename.endswith('.wav'):
            original_file = os.path.join(original_dir, filename)
            gen_file = os.path.join(gen_dir, filename.replace('.wav', 'gen.wav'))

            if os.path.exists(gen_file):
                # Read audio data
                sr_orig, data_orig = wav.read(original_file)
                sr_gen, data_gen = wav.read(gen_file)

                # Convert to mono if stereo
                if len(data_orig.shape) > 1:
                    data_orig = data_orig[:, 0]
                if len(data_gen.shape) > 1:
                    data_gen = data_gen[:, 0]

                # Resample if needed
                if sr_orig != TARGET_SAMPLERATE:
                    num_samples = round(len(data_orig) * TARGET_SAMPLERATE / sr_orig)
                    data_orig = signal.resample(data_orig, num_samples)
                if sr_gen != TARGET_SAMPLERATE:
                    num_samples = round(len(data_gen) * TARGET_SAMPLERATE / sr_gen)
                    data_gen = signal.resample(data_gen, num_samples)

                # Crop both to the shorter length
                min_length = min(len(data_orig), len(data_gen))
                data_orig = data_orig[:min_length]
                data_gen = data_gen[:min_length]

                # Compute FFTs from data directly
                freq1, fft1 = compute_fft_from_data(data_orig, TARGET_SAMPLERATE)
                freq2, fft2 = compute_fft_from_data(data_gen, TARGET_SAMPLERATE)

                # Calculate MSE
                mse = calculate_mse(freq1, freq2, fft1, fft2)

                mse_values_by_freq.append((mse[0], mse[1]))

    return mse_values_by_freq



def average_values_by_label(array_of_2d_arrays):
    sums = defaultdict(float)
    counts = defaultdict(int)

    for arr in array_of_2d_arrays:
        labels = arr[0]   # first row: labels
        values = arr[1]   # second row: values
        
        for label, value in zip(labels, values):
            if label > 10000 or label < 2000:
                continue
            sums[label] += value
            counts[label] += 1

    # Compute average per label
    averages = {label: sums[label]/counts[label] for label in sums}

    return averages


def plot_mse_by_frequency(mse_values_by_freq):
    avg_mse = average_values_by_label(mse_values_by_freq)
    plt.figure(figsize=(14, 6))
    labels = list(avg_mse.keys())
    values = list(avg_mse.values())
    order = np.argsort(labels)  # indices that sort the labels
    
    labels = np.array(labels)[order]
    values = np.array(values)[order]
    plt.plot(labels,values)
    plt.xlabel('Frequency (Hz)',  fontsize=16)
    plt.ylabel('MSE',  fontsize=16)
    plt.title('22.7ms Chunks',  fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.savefig("error_by_freq_22ms.pdf")   # Save the figure to a file
    plt.show()

def main(base_dir=CODA_DIR, evaluation_dir=REGEN_CODA_DIR):
    """Calculate average mse error between both sets of audio files accross each frequency band
    Parameters:
        base_dir: Baseline audio path
        evaluation_dir: Audio files to compare against
    """
    mse_values_by_freq = process_files(base_dir, evaluation_dir)
    plot_mse_by_frequency(mse_values_by_freq)

if __name__ == "__main__":
    main()
