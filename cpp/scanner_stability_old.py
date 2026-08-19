import argparse
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
import matplotlib.dates as mdates

# ========= PARAMETERS =========
BASELINE_START, BASELINE_END = 100, 200
N_PEAKS = 4
# ==============================

parser = argparse.ArgumentParser(description="DAPHNE 5-Day LED Symmetric Relative Analysis")
parser.add_argument(
    "--runs",
    type=str,
    required=True,
    help="Run numbers representing different times, e.g., 1719,1720,1721,1722,1723",
)
parser.add_argument(
    "--afes",
    type=str,
    default="4,3,2,1,0",
    help="AFEs to analyze, default: 4,3,2,1,0"
)
parser.add_argument(
    "--chs",
    type=str,
    default="0,2,5,7",
    help="Channels to analyze, default: 0,2,5,7"
)
args = parser.parse_args()

run_list = sorted([int(r.strip()) for r in args.runs.split(",") if r.strip()])
afe_list = sorted([int(a.strip()) for a in args.afes.split(",") if a.strip()])
ch_list = sorted([int(ch.strip()) for ch in args.chs.split(",") if ch.strip()])

history_data = {}

for run in run_list:
    for afe in afe_list:
        for ch in ch_list:
            pattern = f"data/run{run}_afe{afe}_ch{ch}_*.hdf5"
            matched_files = glob(pattern)

            if not matched_files:
                print(f"Skip Run{run} AFE{afe} CH{ch} (File not found)")
                continue

            filename = matched_files[0]

            try:
                base_name = os.path.basename(filename)
                date_str = base_name.replace(".hdf5", "").split("_")[-2]
                file_date = datetime.strptime(date_str, "%Y%m%d")
            except Exception as e:
                print(f"Warning: Could not parse date from {filename}: {e}")
                continue

            try:
                with h5py.File(filename, "r") as f:
                    wvfms = np.array(f["data"], dtype=np.float64)
                    avg_wvfm = wvfms.mean(axis=0)
                    std_wvfm = wvfms.std(axis=0)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # ===== Baseline Calibration =====
            avg_wvfm = -avg_wvfm
            baseline = avg_wvfm[BASELINE_START:BASELINE_END].mean()
            adjusted = avg_wvfm - baseline

            # ===== Peak Finding =====
            peaks, _ = find_peaks(adjusted, height=500)

            if len(peaks) < N_PEAKS:
                print(f"Warning: File {base_name} found fewer than {N_PEAKS} peaks. Skipping.")
                continue

            # Sort by amplitude, select the largest N_PEAKS
            peak_heights = adjusted[peaks]
            idx_sorted = np.argsort(peak_heights)[-N_PEAKS:]
            selected_peaks = np.sort(peaks[idx_sorted])

            amps = adjusted[selected_peaks]
            errs = std_wvfm[selected_peaks]

            # ===== Classify into History Records =====
            for peak_idx, (amp, err) in enumerate(zip(amps, errs)):
                key = (afe, ch, peak_idx)
                if key not in history_data:
                    history_data[key] = []
                history_data[key].append((file_date, amp, err))

# ===== Create 1-Row, 3-Column Figure Canvas =====
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

all_mean_daily_changes = []  # 收集每个 LED 的平均每日对称绝对变化率
all_symmetric_spreads = []   # 收集每个 LED 的 (max-min)/((max+min)/2)
led_labels = []

# Loop for calculations and plotting trends
for (afe, ch, peak_idx), trend in history_data.items():
    trend = sorted(trend, key=lambda x: x[0])
    dates = [item[0] for item in trend]
    amps = [item[1] for item in trend]
    errs = [item[2] for item in trend]

    if len(amps) < 2:
        continue

    # 1. 💡 图 2 核心计算：每一天相对于前一天的 (x - y) / ((x + y) / 2) 并求绝对值平均数
    daily_changes = []
    for i in range(len(amps) - 1):
        day_start = amps[i]
        day_end = amps[i+1]
        denominator = (day_end + day_start) / 2.0
        if denominator != 0:
            # 计算相邻两天的对称绝对变化率
            change = (abs(day_end - day_start) / denominator) * 100.0
            daily_changes.append(change)
            
    mean_daily_change = np.mean(daily_changes) if daily_changes else 0
    all_mean_daily_changes.append(mean_daily_change)
    led_labels.append(f"AFE{afe}_CH{ch}_P{peak_idx}")

    # 2. 💡 图 3 核心计算：(max - min) / ((max + min) / 2)
    max_amp = max(amps)
    min_amp = min(amps)
    spread_denominator = (max_amp + min_amp) / 2.0
    if spread_denominator != 0:
        symmetric_spread = ((max_amp - min_amp) / spread_denominator) * 100.0
        all_symmetric_spreads.append(symmetric_spread)

    # 3. Plot 1: Scanner Stability (Amplitudes over Time with Error Bars)
    ax1.errorbar(
        dates,
        amps,
        yerr=errs,
        marker="o",
        linestyle="-",
        alpha=0.6,
        capsize=3,
        elinewidth=1,
        label=f"AFE{afe}_CH{ch}_P{peak_idx} (Daily Δ: {mean_daily_change:.2f}%)",
    )

# --- Plot 1 Formatting ---
ax1.set_title("Scanner Stability", fontsize=12)
ax1.set_xlabel("Time", fontsize=11)
ax1.set_ylabel("Amplitude [ADC]", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(0, 8000)

# --- Plot 2 Formatting (Histogram of Mean Symmetric Daily Changes) ---
chg_bins = np.linspace(0, max(max(all_mean_daily_changes, default=1.0) * 1.2, 5.0), 20)
ax2.hist(all_mean_daily_changes, bins=chg_bins, color="lightblue", edgecolor="black", alpha=0.8)
ax2.set_title("Distribution of Mean Absolute Daily Change", fontsize=12)
ax2.set_xlabel("Mean Symmetric Daily Change |x-y|/((x+y)/2) [%]", fontsize=11)
ax2.set_ylabel("Number of LEDs", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5)

# --- Plot 3 Formatting (Histogram of Symmetric Max-Min Spreads) ---
spread_bins = np.linspace(0, max(max(all_symmetric_spreads, default=1.0) * 1.2, 5.0), 20)
ax3.hist(all_symmetric_spreads, bins=spread_bins, color="lightcoral", edgecolor="black", alpha=0.8)
ax3.set_title("Distribution of Relative Max-Min Spread", fontsize=12)
ax3.set_xlabel("Symmetric Spread (max-min)/((max+min)/2) [%]", fontsize=11)
ax3.set_ylabel("Number of LEDs", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.5)

fig.autofmt_xdate(rotation=30)
plt.tight_layout()

# ==========================================================
# Terminal Diagnostic Console Report
# ==========================================================
print("\n" + "="*60)
print(f"📊 DAPHNE 5-Day Data Screening Report (Total: {len(led_labels)} Channels)")
print("="*60)
if len(all_mean_daily_changes) > 0:
    sorted_idx = np.argsort(all_mean_daily_changes)[::-1]
    print("🚨 Warning: Top 5 channels experiencing the highest mean daily instability:")
    for i in range(min(5, len(sorted_idx))):
        t_idx = sorted_idx[i]
        print(f"  Rank [{i+1}] -> {led_labels[t_idx]:<16} | Mean Daily Abs Change: {all_mean_daily_changes[t_idx]:.2f}% | Relative Spread: {all_symmetric_spreads[t_idx]:.2f}%")
print("="*60 + "\n")

plt.show()
