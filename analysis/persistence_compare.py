import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import argparse
from tqdm import tqdm
import os

def extract_amplitudes(filename, range_start=250, range_end=750, pedestal_range=(0, 150), window_size=1):
    with h5py.File(filename, 'r') as f:
        wvfms = np.array(f['data']).astype('float')

    wvfms = wvfms[:, range_start:range_end]
    amplitudes = []

    for i in range(wvfms.shape[0]):
        wvfm = wvfms[i]
        mean = np.mean(wvfm[pedestal_range[0]:pedestal_range[1]])
        df = pd.DataFrame({'wvfm': wvfm})
        smoothed = df['wvfm'].rolling(window=window_size).mean()
        smoothed = smoothed - mean
        amplitudes.append(np.min(smoothed))

    return amplitudes

def plot_4channel_comparison(files_off, files_on, bins=75, range_=(-1000, 100)):
    ch_labels = ["CH0", "CH2", "CH5", "CH7"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axes = axes.flatten()

    for i in range(4):
        amp_off = extract_amplitudes(files_off[i])
        amp_on = extract_amplitudes(files_on[i])

        axes[i].hist(amp_off, bins=bins, range=range_, alpha=0.6, color='blue', label='Lights Off')
        axes[i].hist(amp_on, bins=bins, range=range_, alpha=0.6, color='green', label='Latches Fastened')

        axes[i].set_title(f"{ch_labels[i]}")
        axes[i].set_xlabel("Amplitude")
        axes[i].set_ylabel("Counts")
        axes[i].legend()
        axes[i].grid(False)

    plt.suptitle("Noise Comparison Across Channels", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare amplitude distributions of CH0, CH2, CH5, CH7")
    parser.add_argument("files", nargs=8, help="Order: ch0_off ch0_on ch2_off ch2_on ch5_off ch5_on ch7_off ch7_on")
    args = parser.parse_args()

    files_off = [args.files[0], args.files[2], args.files[4], args.files[6]]
    files_on  = [args.files[1], args.files[3], args.files[5], args.files[7]]

    plot_4channel_comparison(files_off, files_on)
