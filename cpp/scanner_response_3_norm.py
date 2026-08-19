import argparse
from glob import glob

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Configuration
# ==========================================================
BASELINE_START, BASELINE_END = 100, 200

# QC acceptance criterion:
# Pass if |deviation| <= 25%
# Fail if |deviation| > 25%
THRESHOLD = 25.0

# Maximum deviation represented by the color scale.
# Values above 50% use the same darkest red.
MAX_DISPLAY_DEVIATION = 50.0

# Four pulse windows in each supercell
WINDOWS = [
    (600, 900),
    (1500, 1900),
    (2400, 2800),
    (3300, 3700),
]


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Module Response 3: "
            "Absolute Normalized QC Heatmap"
        )
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Test run number, for example: 2003",
    )

    parser.add_argument(
        "--ref",
        type=int,
        required=True,
        help="Reference run number, for example: 2000",
    )

    parser.add_argument(
        "--afes",
        type=str,
        default="0,1,2,3,4",
        help="AFEs to analyze, default: 0,1,2,3,4",
    )

    parser.add_argument(
        "--chs",
        type=str,
        default="0,2,5,7",
        help="Channels to analyze, default: 0,2,5,7",
    )

    args = parser.parse_args()

    # Always sort AFEs in ascending order:
    # AFE 0 at the top and AFE 4 at the bottom.
    args.afe_list = sorted(
        int(afe.strip())
        for afe in args.afes.split(",")
        if afe.strip().isdigit()
    )

    args.ch_list = sorted(
        int(ch.strip())
        for ch in args.chs.split(",")
        if ch.strip().isdigit()
    )

    return args


def extract_afe_ch_signal(run, afe, ch):
    """Extract the four pulse amplitudes for one run, AFE, and channel."""

    pattern = f"data/run{run}_afe{afe}_ch{ch}_*.hdf5"
    matched_files = glob(pattern)

    if not matched_files:
        return None

    filename = matched_files[0]

    try:
        with h5py.File(filename, "r") as h5_file:
            waveforms = np.asarray(
                h5_file["data"],
                dtype=np.float64,
            )

            average_waveform = waveforms.mean(axis=0)

    except Exception as error:
        print(f"Error reading {filename}: {error}")
        return None

    average_waveform = -average_waveform

    baseline = average_waveform[
        BASELINE_START:BASELINE_END
    ].mean()

    adjusted_waveform = average_waveform - baseline

    amplitudes = []

    for start, end in WINDOWS:
        segment = adjusted_waveform[start:end]

        local_peak_index = np.argmax(segment)
        amplitude = segment[local_peak_index]

        amplitudes.append(float(amplitude))

    return amplitudes


def main():
    args = parse_args()

    number_of_rows = len(args.afe_list)
    number_of_columns = len(args.ch_list) * 4

    deviation_matrix = np.full(
        (number_of_rows, number_of_columns),
        np.nan,
    )

    print()
    print("=" * 85)

    print(
        f"DAPHNE Supercell Diagnostic Report | "
        f"Run {args.run} vs Reference Run {args.ref}"
    )

    print("=" * 85)

    supercell_count = 0

    for row_index, afe in enumerate(args.afe_list):
        for channel_index, channel in enumerate(args.ch_list):

            test_amplitudes = extract_afe_ch_signal(
                args.run,
                afe,
                channel,
            )

            reference_amplitudes = extract_afe_ch_signal(
                args.ref,
                afe,
                channel,
            )

            if (
                test_amplitudes is None
                or reference_amplitudes is None
            ):
                print(
                    f"Skipping [AFE {afe} | CH {channel}]: "
                    f"data file not found."
                )

                print("-" * 85)
                continue

            supercell_count += 1

            test_amplitudes = np.asarray(
                test_amplitudes,
                dtype=float,
            )

            reference_amplitudes = np.asarray(
                reference_amplitudes,
                dtype=float,
            )

            reference_mean = np.mean(
                reference_amplitudes
            )

            test_mean = np.mean(
                test_amplitudes
            )

            if test_mean == 0:
                print(
                    f"Warning: The mean pulse height is zero for "
                    f"Run {args.run}, AFE {afe}, CH {channel}. "
                    f"Skipping."
                )

                print("-" * 85)
                continue

            # Normalize the test-run amplitudes using the mean
            # amplitude of the four pulses.
            scale_factor = reference_mean / test_mean

            normalized_test_amplitudes = (
                test_amplitudes * scale_factor
            )

            # Relative deviation between normalized test amplitudes
            # and reference amplitudes.
            deviation_percent = (
                normalized_test_amplitudes
                / reference_amplitudes
                - 1.0
            ) * 100.0

            print(
                f"Supercell #{supercell_count:02d} "
                f"[AFE {afe} | CH {channel}]"
            )

            print(
                "  Reference pulse heights: "
                + ", ".join(
                    f"{value:7.1f}"
                    for value in reference_amplitudes
                )
                + f" | Mean: {reference_mean:.1f}"
            )

            print(
                "  Test-run pulse heights:  "
                + ", ".join(
                    f"{value:7.1f}"
                    for value in test_amplitudes
                )
                + f" | Mean: {test_mean:.1f}"
            )

            print(
                f"  Scale factor: {scale_factor:.4f}"
            )

            print(
                "  Normalized test pulses: "
                + ", ".join(
                    f"{value:7.1f}"
                    for value in normalized_test_amplitudes
                )
            )

            print(
                "  Relative deviations:    "
                + ", ".join(
                    f"{value:+.2f}%"
                    for value in deviation_percent
                )
            )

            qc_results = [
                "PASS"
                if abs(value) <= THRESHOLD
                else "FAIL"
                for value in deviation_percent
            ]

            print(
                "  QC results:             "
                + ", ".join(qc_results)
            )

            print("-" * 85)

            start_column = channel_index * 4
            end_column = start_column + 4

            deviation_matrix[
                row_index,
                start_column:end_column,
            ] = deviation_percent

    print(
        f"Analysis complete. Successfully processed "
        f"{supercell_count} supercell channels."
    )

    print("=" * 85)
    print()

    # ==========================================================
    # Calculate QC results
    # ==========================================================
    valid_data = np.isfinite(deviation_matrix)

    pass_matrix = (
        valid_data
        & (np.abs(deviation_matrix) <= THRESHOLD)
    )

    fail_matrix = (
        valid_data
        & (np.abs(deviation_matrix) > THRESHOLD)
    )

    number_of_passes = np.sum(pass_matrix)
    number_of_failures = np.sum(fail_matrix)

    print(
        f"QC result: {number_of_passes} pulse positions passed, "
        f"{number_of_failures} pulse positions failed."
    )

    # Use absolute deviations to determine the displayed color.
    # The signed deviations are still printed inside the cells.
    absolute_deviation_matrix = np.abs(
        deviation_matrix
    )

    # Clip only the displayed colors.
    # The text inside each cell still shows the real value.
    plot_matrix = np.clip(
        absolute_deviation_matrix,
        0.0,
        MAX_DISPLAY_DEVIATION,
    )

    # ==========================================================
    # Plot
    # ==========================================================
    figure, axes = plt.subplots(
        number_of_rows,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.3},
    )

    if number_of_rows == 1:
        axes = [axes]

    # ==========================================================
    # Colorblind-friendly color map
    #
    #  0% to 25%: dark blue -> light blue
    #  >25%:      light red  -> dark red
    #
    # There is a sharp color transition at the 25% QC limit.
    # ==========================================================
    threshold_position = (
        THRESHOLD / MAX_DISPLAY_DEVIATION
    )

    color_definition = [
        # Pass region
        (0.00, "#005A8D"),
        (
            threshold_position * 0.50,
            "#3B8FC2",
        ),
        (
            threshold_position,
            "#A6CEE3",
        ),

        # Sharp transition from Pass to Fail
        (
            threshold_position + 0.0001,
            "#F4A582",
        ),

        # Fail region
        (0.75, "#D6604D"),
        (1.00, "#B2182B"),
    ]

    color_map = (
        mcolors.LinearSegmentedColormap.from_list(
            "Blue_Pass_Red_Fail",
            color_definition,
            N=256,
        )
    )

    # Missing data are shown in white.
    color_map.set_bad(color="white")

    normalization = mcolors.Normalize(
        vmin=0.0,
        vmax=MAX_DISPLAY_DEVIATION,
    )

    image = None

    # args.afe_list is sorted as [0, 1, 2, 3, 4].
    # axes[0] is the top subplot, so AFE 0 appears at the top.
    for row_index, afe in enumerate(args.afe_list):
        axis = axes[row_index]

        row_data = np.ma.masked_invalid(
            np.atleast_2d(
                plot_matrix[row_index, :]
            )
        )

        image = axis.imshow(
            row_data,
            cmap=color_map,
            norm=normalization,
            aspect="auto",
            extent=[
                -0.5,
                number_of_columns - 0.5,
                -0.5,
                0.5,
            ],
        )

        axis.set_yticks([])

        axis.set_ylabel(
            f"AFE {afe}",
            fontsize=12,
            rotation=0,
            horizontalalignment="right",
            verticalalignment="center",
        )

        for column in range(number_of_columns):
            value = deviation_matrix[
                row_index,
                column,
            ]

            if np.isnan(value):
                axis.text(
                    column,
                    0,
                    "NaN",
                    color="gray",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                )

                continue

            axis.text(
                column,
                0,
                f"{value:+.1f}%",
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                fontweight="bold",
            )

        axis.axhline(
            -0.5,
            color="gray",
            linewidth=0.5,
        )

        axis.axhline(
            0.5,
            color="gray",
            linewidth=0.5,
        )

        for column in range(number_of_columns + 1):
            if column % 4 == 0:
                axis.axvline(
                    column - 0.5,
                    color="black",
                    linewidth=2.5,
                )

            else:
                axis.axvline(
                    column - 0.5,
                    color="gray",
                    linewidth=0.5,
                    linestyle="--",
                )

    # ==========================================================
    # Pulse labels
    # ==========================================================
    bottom_axis = axes[-1]

    bottom_axis.set_xticks(
        np.arange(number_of_columns)
    )

    bottom_axis.set_xticklabels(
        [
            f"P{index % 4}"
            for index in range(number_of_columns)
        ],
        fontsize=12,
    )

    # ==========================================================
    # Channel labels
    # ==========================================================
    for channel_index, channel in enumerate(
        args.ch_list
    ):
        bottom_axis.text(
            channel_index * 4 + 1.5,
            -0.9,
            f"CH {channel}",
            color="black",
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="top",
        )

    figure.subplots_adjust(
        bottom=0.18,
        top=0.88,
        left=0.08,
        right=0.84,
    )

    # ==========================================================
    # Colorbar
    # ==========================================================
    colorbar_axis = figure.add_axes(
        [0.88, 0.25, 0.018, 0.55]
    )

    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Absolute Normalization Deviation (%)",
        fontsize=12,
        labelpad=10,
    )

    colorbar.set_ticks(
        [
            0,
            10,
            20,
            THRESHOLD,
            35,
            MAX_DISPLAY_DEVIATION,
        ]
    )

    colorbar.ax.set_yticklabels(
        [
            "0%",
            "10%",
            "20%",
            f"{THRESHOLD:.0f}% — Pass limit",
            "35%",
            f"≥{MAX_DISPLAY_DEVIATION:.0f}%",
        ]
    )

    # Add a line at the 25% acceptance limit.
    colorbar.ax.axhline(
        THRESHOLD,
        color="black",
        linewidth=2.0,
    )

    figure.suptitle(
        f"Module Response 3\n\n"
        f"Run {args.run} vs Reference Run {args.ref}",
        fontsize=14,
    )

    print(
        "The normalized QC heatmap was generated successfully."
    )

    plt.show()


if __name__ == "__main__":
    main()
