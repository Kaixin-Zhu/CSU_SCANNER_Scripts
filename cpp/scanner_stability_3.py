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

parser = argparse.ArgumentParser(description="DAPHNE LED Multi-Day Normalized Relative Analysis")
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

# ===== Create 1-Row, 2-Column Figure Canvas =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 使用字典按“日期”归类收集非第一天的相对偏差数据
ratios_by_date = {}
led_labels = []

# Loop for calculations and plotting trends
for (afe, ch, peak_idx), trend in history_data.items():
    trend = sorted(trend, key=lambda x: x[0])
    
    if len(trend) < 2:
        continue

    dates = [item[0] for item in trend]
    amps = [item[1] for item in trend]
    errs = [item[2] for item in trend]

    ref_amp = amps[0]
    ref_err = errs[0]

    if ref_amp == 0:
        continue

    # ⭐ 核心修改 1：计算相对偏差百分比 (Run/Ref - 1) * 100
    normalized_deviations = [(amp / ref_amp - 1.0) * 100.0 for amp in amps]
    
    # 相对偏差的绝对误差传递公式：d(A/Ref - 1)*100 = 100 * (A/Ref) * sqrt((dA/A)^2 + (dRef/Ref)^2)
    normalized_errs = []
    for amp, err in zip(amps, errs):
        if amp == 0:
            normalized_errs.append(0.0)
        else:
            rel_err = np.sqrt((err / amp)**2 + (ref_err / ref_amp)**2)
            normalized_errs.append(100.0 * (amp / ref_amp) * rel_err)

    # ⭐ 核心修改 2：把后续日期的【相对偏差值】归类进去（跳过第一天的 0.0%）
    for d, dev in zip(dates[1:], normalized_deviations[1:]):
        date_label = d.strftime("%Y-%m-%d")
        if date_label not in ratios_by_date:
            ratios_by_date[date_label] = []
        ratios_by_date[date_label].append(dev)
        
    led_labels.append(f"AFE{afe}_CH{ch}_P{peak_idx}")

    # --- Plot 1: Normalized Scanner Stability (Relative Deviation over Time) ---
    ax1.errorbar(
        dates,
        normalized_deviations,
        yerr=normalized_errs,
        marker="o",
        linestyle="-",
        alpha=0.6,
        capsize=3,
        elinewidth=1,
        label=f"AFE{afe}_CH{ch}_P{peak_idx}",
    )

# --- Plot 1 Formatting ---
ax1.set_title("Normalized Scanner Stability (Relative Deviation)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Time", fontsize=11)
ax1.set_ylabel("Relative Deviation from Reference (%)", fontsize=11)  # 修改单位标签
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.axhline(0.0, color="black", linestyle="--", alpha=0.7, label="Reference Base (0.0%)")  # 基准改为 0%

# --- Plot 2 Formatting (分色多组直方图绘制) ---
all_deviations = [d for d_list in ratios_by_date.values() for d in d_list]
if all_deviations:
    # 🌟 自动根据偏差百分比的范围动态精细分 Bins
    vmin, vmax = min(all_deviations), max(all_deviations)
    ratio_bins = np.linspace(max(vmin - 0.5, -10.0), min(vmax + 0.5, 10.0), 40)
else:
    ratio_bins = np.linspace(-5.0, 5.0, 40)

colors_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
sorted_dates = sorted(ratios_by_date.keys())

hist_data_list = [ratios_by_date[d] for d in sorted_dates]
hist_colors = colors_palette[:len(sorted_dates)]
hist_labels = [f"Run Data on {d}" for d in sorted_dates]

if hist_data_list:
    ax2.hist(hist_data_list, bins=ratio_bins, color=hist_colors, label=hist_labels,
             edgecolor="black", alpha=0.8, histtype='bar')

ax2.set_title("Distribution of All Relative Deviations", fontsize=12, fontweight="bold")
ax2.set_xlabel("Relative Deviation from Reference (%)", fontsize=11)  # 修改轴标签
ax2.set_ylabel("Counts (All Selected Channels)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.axvline(0.0, color="red", linestyle="-", alpha=0.6, label="Ideal Line (0.0%)")  # 基准线改为 0%
ax2.legend()

fig.autofmt_xdate(rotation=30)
plt.tight_layout()

# ==========================================================
# Terminal Diagnostic Console Report
# ==========================================================
print("\n" + "="*60)
print(f"📊 DAPHNE Multi-Day Normalized Relative Report (Total: {len(led_labels)} Channels)")
print("="*60)
for d in sorted_dates:
    day_data = ratios_by_date[d]
    print(f"📅 Date: {d} | Samples: {len(day_data):<3} | Mean Deviation: {np.mean(day_data):+.3f}% | Std: {np.std(day_data):.3f}%")
print("="*60 + "\n")

plt.show()
