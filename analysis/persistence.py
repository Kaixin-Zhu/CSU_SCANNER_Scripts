import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import argparse
from tqdm import tqdm
import pandas as pd
import h5py
from scipy.optimize import curve_fit

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def main(filename):
    if '.csv' in filename:
        wvfms = np.genfromtxt(filename, delimiter=' ', max_rows=10000)
    elif '.hdf5' in filename:
        with h5py.File(filename, 'r') as f:
            wvfms = np.array(f['data']).astype('float')
    else:
        print('AAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHH')
        raise Exception('File type not recognized')

    range_x = np.array([250, 750])
    wvfms = wvfms[:, range_x[0]:range_x[1]]
    num_time_steps = range_x[1] - range_x[0]
    range_x = np.array([0, range_x[1]-range_x[0]])*16

    window_size = 1
    pedestal_range = (0, 150)
    amplitudes = []

    for eventID in tqdm(range(wvfms.shape[0])):
        wvfm = wvfms[eventID]
        mean = np.mean(wvfm[pedestal_range[0]:pedestal_range[1]])
        df = pd.DataFrame({'wvfm': wvfm})
        wvfm_smoothed = df['wvfm'].rolling(window=window_size).mean()
        wvfm_smoothed = wvfm_smoothed - mean
        amplitudes.append(np.min(wvfm_smoothed))
        wvfms[eventID, :] = wvfm_smoothed

    num_waveforms, num_time_steps = wvfms.shape
    wvfms = wvfms.ravel()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    meanADC = np.mean(wvfms[~np.isnan(wvfms)])
    print(f'meanADC = {meanADC}')

    ymax = 4000
    ymin = -8000
    num_bins_y = int((ymax - ymin)/10)
    range_x = [0, num_time_steps * 16]
    range_y = [ymin, ymax]

    plot2d = axes[0].hist2d(
        np.tile(np.arange(num_time_steps)*16, num_waveforms),
        wvfms,
        bins=[150, num_bins_y],
        range=[range_x, range_y],
        cmap='viridis',
        vmin=1,
        vmax=50
    )
    axes[0].set_xlabel(r'Time [ns]')
    axes[0].set_ylabel('Amplitude (ADC counts)')
    axes[0].set_title("Persistence Plot")
    colorbar = fig.colorbar(plot2d[3], ax=axes[0])

    # Histogram of amplitudes
    counts, bin_edges, _ = axes[1].hist(amplitudes, bins=100, range=(-8000, 100), alpha=0.6, label='Data')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess
    p0 = [np.max(counts), np.mean(amplitudes), np.std(amplitudes)]

    try:
        popt, pcov = curve_fit(gauss, bin_centers, counts, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
        y_fit = gauss(x_fit, *popt)
        axes[1].plot(x_fit, y_fit, 'r--', label=f'Gauss Fit\nμ={mu_fit:.1f}, σ={sigma_fit:.1f}')
        print(f"Gaussian Fit: mean = {mu_fit:.3f}, sigma = {sigma_fit:.3f}")
    except RuntimeError:
        print("Gaussian fit failed!")
        mu_fit, sigma_fit = np.nan, np.nan

    axes[1].set_xlabel('Waveform Amplitude')
    axes[1].set_title("Amplitude Distribution")
    axes[1].set_ylim(0, 1000)
    axes[1].legend()
    plt.suptitle("1st Module, CH 7 Signal Analysis")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make RMS distributions")
    parser.add_argument('filepath', help='Input filepath')
    args = parser.parse_args()
    main(args.filepath)
