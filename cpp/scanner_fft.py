#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = "data"
QC_THRESHOLD_DB = 20.0

# QC frequency range in MHz
QC_FREQ_MIN_MHZ = 0.01
QC_FREQ_MAX_MHZ = None

# ==========================================================


class FFTProcessor:
    """Calculate the FFT of a single waveform."""

    def __init__(self, signal, dt=16e-9):
        signal = np.asarray(
            signal,
            dtype=np.float64,
        )

        # Ensure the signal length is even
        if signal.shape[-1] % 2 != 0:
            signal = signal[..., :-1]

        signal_length = signal.shape[-1]

        # Calculate and normalize the FFT
        signal_fft = (
            np.fft.fft(signal) / signal_length
        )

        frequency = np.fft.fftfreq(
            signal_length,
            d=dt,
        )

        # Keep the non-negative frequency side
        positive_mask = frequency >= 0

        positive_frequency = frequency[
            positive_mask
        ]

        positive_fft = signal_fft[
            positive_mask
        ].copy()

        # Convert to a one-sided FFT amplitude
        if positive_fft.size > 1:
            positive_fft[1:] *= 2.0

        # Frequency in MHz
        self.x = positive_frequency / 1e6

        # Amplitude in dBFS
        fft_amplitude = (
            np.abs(positive_fft) / (2**14)
        )

        self.y = 20.0 * np.log10(
            np.maximum(
                fft_amplitude,
                np.finfo(float).tiny,
            )
        )


class MeanFFTAnalyzer:
    """Calculate the mean FFT across multiple waveforms."""

    def __init__(self, data):
        fft_x_values = []
        fft_y_values = []
        waveform_rms_values = []

        for waveform in data:
            result = FFTProcessor(waveform)

            fft_x_values.append(result.x)
            fft_y_values.append(result.y)
            waveform_rms_values.append(
                np.std(waveform)
            )

        self.x = np.mean(
            fft_x_values,
            axis=0,
        )

        self.y = np.mean(
            fft_y_values,
            axis=0,
        )

        self.avg_rms = np.mean(
            waveform_rms_values
        )


def parse_number_list(value, argument_name):
    """Convert a comma-separated string to integers."""
    try:
        numbers = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]

    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain "
            "comma-separated integers."
        ) from error

    if not numbers:
        raise argparse.ArgumentTypeError(
            f"{argument_name} cannot be empty."
        )

    return numbers


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DAPHNE waveform FFT analyzer with a "
            "user-specified reference + 20 dB QC threshold."
        )
    )

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help=(
            "Test runs separated by commas, "
            "e.g. 2008,2009"
        ),
    )

    parser.add_argument(
        "--ref",
        type=int,
        required=True,
        help=(
            "Reference run used to define the "
            "QC threshold"
        ),
    )

    parser.add_argument(
        "--afe",
        type=str,
        default="0,1,2,3,4",
        help=(
            "AFE numbers separated by commas "
            "(default: 0,1,2,3,4)"
        ),
    )

    parser.add_argument(
        "--ch",
        type=str,
        default="0,2,5,7",
        help=(
            "Channel numbers separated by commas "
            "(default: 0,2,5,7)"
        ),
    )

    parser.add_argument(
        "--label",
        type=str,
        default="",
        help=(
            "Optional labels corresponding to the "
            "test runs, separated by commas"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=(
            f"Directory containing HDF5 files "
            f"(default: {DATA_DIR})"
        ),
    )

    args = parser.parse_args()

    try:
        args.runs = parse_number_list(
            args.run,
            "--run",
        )

        args.afes = parse_number_list(
            args.afe,
            "--afe",
        )

        args.channels = parse_number_list(
            args.ch,
            "--ch",
        )

    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    labels = [
        label.strip()
        for label in args.label.split(",")
        if label.strip()
    ]

    args.run_labels = {}

    for index, run_number in enumerate(args.runs):
        if index < len(labels):
            args.run_labels[run_number] = labels[index]
        else:
            args.run_labels[run_number] = (
                f"Run {run_number}"
            )

    return args


def find_run_file(
    data_dir,
    run_number,
    afe_number,
    channel,
):
    """Find the latest matching HDF5 file."""
    pattern = os.path.join(
        data_dir,
        (
            f"run{run_number}_afe{afe_number}"
            f"_ch{channel}_*.hdf5"
        ),
    )

    matching_files = sorted(
        glob.glob(pattern)
    )

    if not matching_files:
        return None

    if len(matching_files) > 1:
        print(
            f"[WARN] Found {len(matching_files)} files for "
            f"Run {run_number}, AFE {afe_number}, "
            f"Ch {channel}; using the latest filename."
        )

    return matching_files[-1]


def load_mean_fft(filename):
    """Load waveforms and calculate the mean FFT."""
    file_size = os.path.getsize(filename)

    if file_size < 2048:
        raise OSError(
            f"File appears incomplete: {filename} "
            f"({file_size} bytes)"
        )

    with h5py.File(filename, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise KeyError(
                f"Dataset 'data' was not found in "
                f"{filename}"
            )

        waveforms = np.asarray(
            hdf5_file["data"],
            dtype=np.float64,
        )

    if waveforms.ndim != 2:
        raise ValueError(
            f"Expected a 2D waveform array, "
            f"got {waveforms.shape}"
        )

    if len(waveforms) == 0:
        raise ValueError(
            f"No waveforms found in {filename}"
        )

    return MeanFFTAnalyzer(waveforms)


def build_qc_mask(frequency):
    """Select the frequency range used for QC."""
    mask = np.isfinite(frequency)

    if QC_FREQ_MIN_MHZ is not None:
        mask &= frequency >= QC_FREQ_MIN_MHZ

    if QC_FREQ_MAX_MHZ is not None:
        mask &= frequency <= QC_FREQ_MAX_MHZ

    return mask


def compare_with_threshold(
    test_frequency,
    test_spectrum,
    reference_frequency,
    reference_spectrum,
):
    """
    Compare the test spectrum with reference + threshold.

    Returns:
        threshold
        qc_mask
        exceed_mask
        maximum_excess_db
    """
    threshold = np.interp(
        test_frequency,
        reference_frequency,
        reference_spectrum + QC_THRESHOLD_DB,
        left=np.nan,
        right=np.nan,
    )

    qc_mask = (
        build_qc_mask(test_frequency)
        & np.isfinite(test_spectrum)
        & np.isfinite(threshold)
    )

    exceed_mask = np.zeros_like(
        test_spectrum,
        dtype=bool,
    )

    exceed_mask[qc_mask] = (
        test_spectrum[qc_mask]
        > threshold[qc_mask]
    )

    if np.any(qc_mask):
        maximum_excess_db = np.max(
            test_spectrum[qc_mask]
            - threshold[qc_mask]
        )
    else:
        maximum_excess_db = np.nan

    return (
        threshold,
        qc_mask,
        exceed_mask,
        maximum_excess_db,
    )


def main():
    args = parse_args()

    test_results = {}
    reference_results = {}

    # ------------------------------------------------------
    # Load reference spectra
    # ------------------------------------------------------

    print(
        f"[INFO] Loading reference spectra "
        f"from Run {args.ref}..."
    )

    for afe_number in args.afes:
        for channel in args.channels:
            reference_filename = find_run_file(
                args.data_dir,
                args.ref,
                afe_number,
                channel,
            )

            if reference_filename is None:
                print(
                    f"[WARN] Missing reference file: "
                    f"Run {args.ref}, "
                    f"AFE {afe_number}, "
                    f"Ch {channel}"
                )
                continue

            try:
                reference_result = load_mean_fft(
                    reference_filename
                )

                reference_results[
                    (afe_number, channel)
                ] = reference_result

                print(
                    f"[INFO] Reference loaded: "
                    f"{os.path.basename(reference_filename)}"
                )

            except Exception as error:
                print(
                    f"[ERROR] Could not process reference "
                    f"file {reference_filename}: {error}"
                )

    if not reference_results:
        print(
            f"[ERROR] No valid reference spectra were "
            f"loaded from Run {args.ref}."
        )
        return

    # ------------------------------------------------------
    # Load test spectra
    # ------------------------------------------------------

    for run_number in args.runs:
        for afe_number in args.afes:
            for channel in args.channels:
                filename = find_run_file(
                    args.data_dir,
                    run_number,
                    afe_number,
                    channel,
                )

                if filename is None:
                    print(
                        f"[WARN] Missing test file: "
                        f"Run {run_number}, "
                        f"AFE {afe_number}, "
                        f"Ch {channel}"
                    )
                    continue

                print(
                    f"[INFO] Analyzing "
                    f"{os.path.basename(filename)}..."
                )

                try:
                    result = load_mean_fft(
                        filename
                    )

                    test_results[
                        (
                            run_number,
                            afe_number,
                            channel,
                        )
                    ] = result

                except Exception as error:
                    print(
                        f"[ERROR] Could not process "
                        f"{filename}: {error}"
                    )

    if not test_results:
        print(
            "[ERROR] No valid test spectra were loaded."
        )
        return

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    color_map = plt.colormaps.get_cmap(
        "rainbow"
    )

    test_items = list(
        test_results.items()
    )

    plotted_thresholds = set()

    for plot_index, (
        (
            run_number,
            afe_number,
            channel,
        ),
        test_result,
    ) in enumerate(test_items):

        if len(test_items) > 1:
            color_ratio = (
                plot_index
                / (len(test_items) - 1)
            )
        else:
            color_ratio = 0.0

        color = color_map(color_ratio)

        run_label = args.run_labels[
            run_number
        ]

        curve_label = (
            f"{run_label}, "
            f"AFE {afe_number}, "
            f"Ch {channel} "
            f"(RMS: {test_result.avg_rms:.2f})"
        )

        axis.plot(
            test_result.x,
            test_result.y,
            color=color,
            linewidth=1.2,
            alpha=0.85,
            label=curve_label,
        )

        reference_key = (
            afe_number,
            channel,
        )

        reference_result = (
            reference_results.get(
                reference_key
            )
        )

        if reference_result is None:
            print(
                f"[WARN] QC not evaluated for "
                f"Run {run_number}, "
                f"AFE {afe_number}, "
                f"Ch {channel}: "
                "reference spectrum is missing."
            )
            continue

        (
            threshold,
            qc_mask,
            exceed_mask,
            maximum_excess_db,
        ) = compare_with_threshold(
            test_result.x,
            test_result.y,
            reference_result.x,
            reference_result.y,
        )

        # Plot one threshold line for each AFE/channel
        if reference_key not in plotted_thresholds:
            axis.plot(
                reference_result.x,
                (
                    reference_result.y
                    + QC_THRESHOLD_DB
                ),
                color="red",
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=(
                    f"Run {args.ref} + "
                    f"{QC_THRESHOLD_DB:.0f} dB "
                    f"(AFE {afe_number}, "
                    f"Ch {channel})"
                ),
            )

            plotted_thresholds.add(
                reference_key
            )

        if np.any(exceed_mask):
            difference = (
                test_result.y - threshold
            )

            masked_difference = np.where(
                qc_mask,
                difference,
                np.nan,
            )

            worst_index = np.nanargmax(
                masked_difference
            )

            worst_frequency = (
                test_result.x[worst_index]
            )

            print(
                f"[QC] FAIL: "
                f"Run {run_number}, "
                f"AFE {afe_number}, "
                f"Ch {channel}; "
                f"maximum excess = "
                f"{maximum_excess_db:.2f} dB "
                f"at {worst_frequency:.4g} MHz"
            )

            axis.scatter(
                test_result.x[exceed_mask],
                test_result.y[exceed_mask],
                color="red",
                s=12,
                zorder=5,
            )

        else:
            print(
                f"[QC] PASS: "
                f"Run {run_number}, "
                f"AFE {afe_number}, "
                f"Ch {channel}; "
                f"maximum excess = "
                f"{maximum_excess_db:.2f} dB"
            )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------

    axis.set_xscale("log")
    axis.set_ylim(-120, 0)

    axis.set_title(
        (
            "DAPHNE Waveform Mean FFT\n"
            f"QC threshold: Run {args.ref} "
            f"reference + {QC_THRESHOLD_DB:.0f} dB"
        ),
        fontsize=14,
    )

    axis.set_xlabel(
        "Frequency [MHz]",
        fontsize=12,
    )

    axis.set_ylabel(
        "Magnitude [dBFS]",
        fontsize=12,
    )

    axis.grid(
        True,
        which="both",
        linestyle="-",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
