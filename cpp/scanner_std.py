#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# Settings
# ============================================================

DATASET_NAME = "data"

BASELINE_START = 100
BASELINE_END = 200

PULSE_WINDOWS = [
    (200, 800),
    (1200, 1800),
    (2200, 2800),
    (3200, 3800),
]

INVERT_SIGN = True
DEFAULT_BINS = 60


# ============================================================
# Gaussian fitting
# ============================================================

def gaussian(x, amplitude, mean, sigma, offset):
    sigma = max(abs(sigma), 1e-12)

    return (
        amplitude
        * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        + offset
    )


def fit_gaussian(values, bins=DEFAULT_BINS):
    """
    Fit a Gaussian distribution.

    Returns
    -------
    mean : float
        Gaussian-fit mean.
    sigma : float
        Gaussian-fit standard deviation.
    """

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size < 10:
        return np.nan, np.nan

    histogram, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    initial_mean = np.mean(values)
    initial_sigma = np.std(values, ddof=1)

    if not np.isfinite(initial_sigma) or initial_sigma <= 0:
        return initial_mean, np.nan

    initial_parameters = [
        histogram.max(),
        initial_mean,
        initial_sigma,
        0.0,
    ]

    try:
        parameters, _ = curve_fit(
            gaussian,
            centers,
            histogram,
            p0=initial_parameters,
            maxfev=20000,
        )

        _, fitted_mean, fitted_sigma, _ = parameters

        return (
            float(fitted_mean),
            abs(float(fitted_sigma)),
        )

    except Exception as error:
        print(f"[WARN] Gaussian fit failed: {error}")

        # Fall back to the sample mean and standard deviation.
        return initial_mean, initial_sigma


# ============================================================
# File discovery
# ============================================================

def extract_afe_channel(filename):
    """
    Extract AFE and channel from a filename such as:

    run1726_afe2_ch5_20260603_104920.hdf5
    """

    filename = Path(filename).name

    match = re.search(
        r"_afe(\d+)_ch(\d+)(?:_|\.hdf5)",
        filename,
    )

    if match is None:
        return None

    afe = int(match.group(1))
    channel = int(match.group(2))

    return afe, channel


def find_run_files(run, data_directory):
    """
    Find all files belonging to one run.

    If multiple files exist for the same AFE/channel,
    the file with the latest timestamp in its name is used.
    """

    data_directory = Path(data_directory)

    pattern = str(
        data_directory
        / f"run{run}_afe*_ch*_*.hdf5"
    )

    filenames = [
        Path(filename)
        for filename in glob.glob(pattern)
    ]

    # Also support filenames without timestamps.
    second_pattern = str(
        data_directory
        / f"run{run}_afe*_ch*.hdf5"
    )

    filenames.extend(
        Path(filename)
        for filename in glob.glob(second_pattern)
    )

    filenames = sorted(set(filenames))

    if not filenames:
        raise FileNotFoundError(
            f"No files were found for Run {run}.\n"
            f"Expected pattern:\n"
            f"  run{run}_afe{{AFE}}_ch{{CH}}_{{timestamp}}.hdf5"
        )

    files_by_position = {}

    for filename in filenames:
        result = extract_afe_channel(filename)

        if result is None:
            print(
                f"[WARN] Cannot extract AFE/channel from "
                f"{filename.name}; skipping."
            )
            continue

        afe, channel = result
        key = (afe, channel)

        if key not in files_by_position:
            files_by_position[key] = filename
        else:
            previous = files_by_position[key]

            # Alphabetical ordering works for timestamps formatted
            # as YYYYMMDD_HHMMSS.
            if filename.name > previous.name:
                print(
                    f"[WARN] Multiple files for AFE {afe}, "
                    f"CH {channel}; using {filename.name}"
                )
                files_by_position[key] = filename

    return files_by_position


# ============================================================
# HDF5 loading
# ============================================================

def load_waveforms(filename):
    filename = Path(filename)

    if filename.stat().st_size < 2048:
        raise OSError(
            f"The file appears incomplete: {filename}\n"
            f"File size: {filename.stat().st_size} bytes"
        )

    with h5py.File(filename, "r") as hdf5_file:
        if DATASET_NAME not in hdf5_file:
            available_datasets = list(hdf5_file.keys())

            raise KeyError(
                f'Dataset "{DATASET_NAME}" was not found in '
                f"{filename}.\n"
                f"Available datasets: {available_datasets}"
            )

        waveforms = np.asarray(
            hdf5_file[DATASET_NAME],
            dtype=np.float64,
        )

    if waveforms.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional waveform array, "
            f"but found {waveforms.shape} in {filename}."
        )

    return waveforms


# ============================================================
# Pulse amplitude extraction
# ============================================================

def extract_pulse_amplitudes(waveforms):
    """
    Extract the peak amplitude in each pulse window.
    """

    processed = waveforms.astype(np.float64)

    if INVERT_SIGN:
        processed = -processed

    waveform_length = processed.shape[1]

    baseline_start = max(
        0,
        min(waveform_length, BASELINE_START),
    )

    baseline_end = max(
        0,
        min(waveform_length, BASELINE_END),
    )

    if baseline_end <= baseline_start:
        raise ValueError(
            "The baseline window is invalid."
        )

    baseline = np.mean(
        processed[:, baseline_start:baseline_end],
        axis=1,
        keepdims=True,
    )

    processed = processed - baseline

    amplitudes_by_pulse = []

    for pulse_number, (start, end) in enumerate(
        PULSE_WINDOWS,
        start=1,
    ):
        start = max(0, min(waveform_length, start))
        end = max(0, min(waveform_length, end))

        if end <= start:
            raise ValueError(
                f"Invalid window for Pulse {pulse_number}: "
                f"({start}, {end})"
            )

        amplitudes = np.max(
            processed[:, start:end],
            axis=1,
        )

        amplitudes = amplitudes[
            np.isfinite(amplitudes)
        ]

        amplitudes_by_pulse.append(amplitudes)

    return amplitudes_by_pulse


def calculate_file_statistics(filename, bins):
    waveforms = load_waveforms(filename)
    amplitudes_by_pulse = extract_pulse_amplitudes(waveforms)

    results = []

    for pulse_number, amplitudes in enumerate(
        amplitudes_by_pulse,
        start=1,
    ):
        mean, sigma = fit_gaussian(
            amplitudes,
            bins=bins,
        )

        if np.isfinite(mean) and mean != 0:
            relative_sigma = (
                sigma / abs(mean) * 100.0
            )
        else:
            relative_sigma = np.nan

        results.append({
            "pulse": pulse_number,
            "mean": mean,
            "sigma": sigma,
            "relative_sigma": relative_sigma,
            "entries": len(amplitudes),
        })

    return results


# ============================================================
# Analyze every common AFE/channel
# ============================================================

def analyze_all_positions(
    run,
    reference_run,
    run_files,
    reference_files,
    bins,
):
    run_positions = set(run_files)
    reference_positions = set(reference_files)

    common_positions = sorted(
        run_positions & reference_positions
    )

    missing_in_reference = sorted(
        run_positions - reference_positions
    )

    missing_in_run = sorted(
        reference_positions - run_positions
    )

    for afe, channel in missing_in_reference:
        print(
            f"[WARN] No reference file for "
            f"AFE {afe}, CH {channel}; skipping."
        )

    for afe, channel in missing_in_run:
        print(
            f"[WARN] No test file for "
            f"AFE {afe}, CH {channel}; skipping."
        )

    if not common_positions:
        raise RuntimeError(
            "The test run and reference run do not have "
            "any common AFE/channel combinations."
        )

    all_results = []

    total = len(common_positions)

    for index, (afe, channel) in enumerate(
        common_positions,
        start=1,
    ):
        run_file = run_files[(afe, channel)]
        reference_file = reference_files[
            (afe, channel)
        ]

        print()
        print(
            f"[INFO] [{index}/{total}] "
            f"AFE {afe}, CH {channel}"
        )
        print(f"       Test: {run_file.name}")
        print(f"       Ref:  {reference_file.name}")

        try:
            run_statistics = calculate_file_statistics(
                run_file,
                bins=bins,
            )

            reference_statistics = (
                calculate_file_statistics(
                    reference_file,
                    bins=bins,
                )
            )

        except Exception as error:
            print(
                f"[ERROR] Failed to analyze "
                f"AFE {afe}, CH {channel}: {error}"
            )
            continue

        for test, reference in zip(
            run_statistics,
            reference_statistics,
        ):
            sigma_difference = (
                test["sigma"]
                - reference["sigma"]
            )

            if (
                np.isfinite(reference["sigma"])
                and reference["sigma"] != 0
            ):
                percent_change = (
                    sigma_difference
                    / reference["sigma"]
                    * 100.0
                )
            else:
                percent_change = np.nan

            all_results.append({
                "afe": afe,
                "channel": channel,
                "pulse": test["pulse"],
                "run_mean": test["mean"],
                "run_sigma": test["sigma"],
                "run_relative_sigma":
                    test["relative_sigma"],
                "reference_mean": reference["mean"],
                "reference_sigma":
                    reference["sigma"],
                "reference_relative_sigma":
                    reference["relative_sigma"],
                "sigma_difference":
                    sigma_difference,
                "percent_change":
                    percent_change,
                "run_file": run_file.name,
                "reference_file":
                    reference_file.name,
            })

    if not all_results:
        raise RuntimeError(
            "No files were analyzed successfully."
        )

    return all_results


# ============================================================
# CSV output
# ============================================================

def save_csv(results, output_filename):
    fieldnames = [
        "afe",
        "channel",
        "pulse",
        "run_mean",
        "run_sigma",
        "run_relative_sigma",
        "reference_mean",
        "reference_sigma",
        "reference_relative_sigma",
        "sigma_difference",
        "percent_change",
        "run_file",
        "reference_file",
    ]

    with open(
        output_filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Saved CSV: {output_filename}")


# ============================================================
# Table plots
# ============================================================

def format_value(value, decimals=1, sign=False):
    if not np.isfinite(value):
        return "N/A"

    if sign:
        return f"{value:+.{decimals}f}"

    return f"{value:.{decimals}f}"


def create_pulse_table(
    results,
    run,
    reference_run,
    pulse_number,
    output_filename,
):
    pulse_results = [
        result
        for result in results
        if result["pulse"] == pulse_number
    ]

    pulse_results.sort(
        key=lambda result: (
            result["afe"],
            result["channel"],
        )
    )

    rows = []

    for result in pulse_results:
        rows.append([
            str(result["afe"]),
            str(result["channel"]),
            format_value(
                result["run_sigma"]
            ),
            format_value(
                result["reference_sigma"]
            ),
            format_value(
                result["sigma_difference"],
                sign=True,
            ),
            (
                format_value(
                    result["percent_change"],
                    sign=True,
                )
                + "%"
            ),
        ])

    column_labels = [
        "AFE",
        "CH",
        f"Run {run}\nσ (ADC)",
        f"Ref {reference_run}\nσ (ADC)",
        "Difference\n(ADC)",
        "Change\n(%)",
    ]

    number_of_rows = len(rows)

    figure_height = max(
        4.5,
        1.5 + 0.42 * number_of_rows,
    )

    figure, axis = plt.subplots(
        figsize=(11, figure_height)
    )

    axis.axis("off")

    table = axis.table(
        cellText=rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)

    number_of_columns = len(column_labels)

    # Header
    for column in range(number_of_columns):
        header = table[(0, column)]
        header.set_facecolor("#C9683B")
        header.set_text_props(
            color="white",
            weight="bold",
        )

    # Body
    for row_index in range(1, number_of_rows + 1):
        if row_index % 2 == 1:
            background = "#F1F1F1"
        else:
            background = "white"

        for column in range(number_of_columns):
            table[(row_index, column)].set_facecolor(
                background
            )

        # Color the difference and change columns.
        table[(row_index, 4)].set_text_props(
            color="#C94720",
            weight="bold",
        )
        table[(row_index, 5)].set_text_props(
            color="#C94720",
            weight="bold",
        )

    start, end = PULSE_WINDOWS[pulse_number - 1]

    axis.set_title(
        f"Pulse {pulse_number} Standard Deviation\n"
        f"Run {run} vs. Reference Run {reference_run} "
        f"(samples {start}–{end})",
        fontsize=16,
        pad=18,
    )

    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"[INFO] Saved Pulse {pulse_number} table: "
        f"{output_filename}"
    )

    plt.close(figure)


def create_summary_table(
    results,
    run,
    reference_run,
    output_filename,
):
    """
    Make one compact table containing σ for all four pulses.
    """

    positions = sorted({
        (result["afe"], result["channel"])
        for result in results
    })

    result_lookup = {
        (
            result["afe"],
            result["channel"],
            result["pulse"],
        ): result
        for result in results
    }

    rows = []

    for afe, channel in positions:
        row = [
            str(afe),
            str(channel),
        ]

        for pulse_number in range(1, 5):
            result = result_lookup.get(
                (afe, channel, pulse_number)
            )

            if result is None:
                row.extend(["N/A", "N/A"])
            else:
                row.extend([
                    format_value(
                        result["run_sigma"]
                    ),
                    format_value(
                        result["reference_sigma"]
                    ),
                ])

        rows.append(row)

    column_labels = [
        "AFE",
        "CH",
        "P1 Test",
        "P1 Ref",
        "P2 Test",
        "P2 Ref",
        "P3 Test",
        "P3 Ref",
        "P4 Test",
        "P4 Ref",
    ]

    figure_height = max(
        5.0,
        1.5 + 0.40 * len(rows),
    )

    figure, axis = plt.subplots(
        figsize=(15, figure_height)
    )

    axis.axis("off")

    table = axis.table(
        cellText=rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.5)

    for column in range(len(column_labels)):
        header = table[(0, column)]
        header.set_facecolor("#C9683B")
        header.set_text_props(
            color="white",
            weight="bold",
        )

    for row_index in range(1, len(rows) + 1):
        background = (
            "#F1F1F1"
            if row_index % 2 == 1
            else "white"
        )

        for column in range(len(column_labels)):
            table[(row_index, column)].set_facecolor(
                background
            )

    axis.set_title(
        f"Gaussian-Fit Standard Deviations, σ (ADC)\n"
        f"Run {run} vs. Reference Run {reference_run}",
        fontsize=17,
        pad=18,
    )

    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"[INFO] Saved summary table: {output_filename}")

    plt.close(figure)


# ============================================================
# Terminal output
# ============================================================

def print_results(results):
    print()
    print(
        f"{'AFE':>4} "
        f"{'CH':>4} "
        f"{'Pulse':>6} "
        f"{'Test σ':>12} "
        f"{'Ref σ':>12} "
        f"{'Difference':>12} "
        f"{'Change':>10}"
    )

    print("-" * 72)

    for result in results:
        print(
            f'{result["afe"]:>4} '
            f'{result["channel"]:>4} '
            f'{result["pulse"]:>6} '
            f'{result["run_sigma"]:>12.1f} '
            f'{result["reference_sigma"]:>12.1f} '
            f'{result["sigma_difference"]:>+12.1f} '
            f'{result["percent_change"]:>+9.1f}%'
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Gaussian-fit standard deviations "
            "for all AFE/channel files and compare them "
            "with a reference run."
        )
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Test run number.",
    )

    parser.add_argument(
        "--ref",
        type=int,
        required=True,
        help="Reference run number.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing the HDF5 files.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of histogram bins.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="std_comparison",
        help="Prefix for output PNG and CSV files.",
    )

    args = parser.parse_args()

    print(f"[INFO] Searching for Run {args.run}")
    run_files = find_run_files(
        args.run,
        args.data_dir,
    )

    print(
        f"[INFO] Found {len(run_files)} "
        f"AFE/channel files for Run {args.run}"
    )

    print(
        f"[INFO] Searching for Reference Run {args.ref}"
    )

    reference_files = find_run_files(
        args.ref,
        args.data_dir,
    )

    print(
        f"[INFO] Found {len(reference_files)} "
        f"AFE/channel files for Reference Run {args.ref}"
    )

    results = analyze_all_positions(
        run=args.run,
        reference_run=args.ref,
        run_files=run_files,
        reference_files=reference_files,
        bins=args.bins,
    )

    print_results(results)

    output_prefix = args.output_prefix

    save_csv(
        results,
        f"{output_prefix}.csv",
    )

    create_summary_table(
        results=results,
        run=args.run,
        reference_run=args.ref,
        output_filename=(
            f"{output_prefix}_summary.png"
        ),
    )

    for pulse_number in range(1, 5):
        create_pulse_table(
            results=results,
            run=args.run,
            reference_run=args.ref,
            pulse_number=pulse_number,
            output_filename=(
                f"{output_prefix}_pulse"
                f"{pulse_number}.png"
            ),
        )

    print()
    print("[DONE] Analysis completed.")


if __name__ == "__main__":
    main()
