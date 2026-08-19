import argparse
import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import warnings

# ==========================================================
# ### CONFIGURATION ###
# ==========================================================
DATA_DIR = "data"  # Directory where the HDF5 files are stored

# 硬编码的特殊 Run 配置：固定使用黑色绘制，名称直接在代码里指定
BLACK_RUNS_CONFIG = {
    1648: "Run 1648, AFE 0, Ch 0 (Bad)",
    1650: "Run 1650, AFE 0, Ch 0 (Bad)",
    1652: "Run 1652, AFE 0, Ch 0 (Bad)",
}
# ==========================================================


class fft_processor:
    """Processes FFT for a single waveform"""

    def __init__(self, sig, dt=16e-9):
        np.seterr(divide="ignore")
        # Ensure the signal length is even
        if sig.shape[-1] % 2 != 0:
            sig = sig[..., :-1]

        t_len = sig.shape[-1]

        # Calculate FFT and normalize
        sigFFT = np.fft.fft(sig) / t_len
        freq = np.fft.fftfreq(t_len, d=dt)

        # Keep only the positive frequency side
        firstNegInd = np.argmax(freq < 0)
        freqAxisPos = freq[:firstNegInd]
        sigFFTPos = 2 * sigFFT[:firstNegInd]

        # Convert to MHz and dBFS
        self.x = freqAxisPos / 1e6
        self.y = 20 * np.log10(np.abs(sigFFTPos) / 2**14)


class mean_fft_analyzer:
    """Calculates the mean FFT across multiple waveforms"""

    def __init__(self, data):
        fft_list_x = []
        fft_list_y = []
        std_list = []

        for k in range(len(data)):
            res = fft_processor(data[k])
            fft_list_x.append(res.x)
            fft_list_y.append(res.y)
            std_list.append(np.std(data[k]))

        self.x = np.mean(fft_list_x, axis=0)
        self.y = np.mean(fft_y := fft_list_y, axis=0)
        self.avg_rms = np.mean(std_list)


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="DAPHNE Waveform FFT Analyzer (Multi-Run, Multi-AFE & Multi-Ch)"
    )

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run numbers, separated by commas (e.g., 1724,1725)",
    )
    parser.add_argument(
        "--afe",
        type=str,
        default="0,1,2,3,4",
        help="AFE indexes, separated by commas (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--ch",
        type=str,
        default="0,2,5,7",
        help="Channels, separated by commas (default: 0,2,5,7)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Cable names for each run, separated by commas (e.g., old,white,green)",
    )

    args = parser.parse_args()

    # Parse multiple Run numbers
    try:
        args.runs = [
            int(r.strip()) for r in args.run.split(",") if r.strip().isdigit()
        ]
        if not args.runs:
            raise ValueError
    except ValueError:
        parser.error(
            "❌ Invalid format for --run. Please enter comma-separated numbers, e.g., 1724,1725"
        )

    # Parse multiple AFE numbers
    try:
        args.afes = [
            int(a.strip()) for a in args.afe.split(",") if a.strip().isdigit()
        ]
        if not args.afes:
            raise ValueError
    except ValueError:
        parser.error("❌ Invalid format for --afe. Please enter comma-separated numbers, e.g., 0,1,2,3,4")

    # Parse multiple Channels
    try:
        args.channels = [
            int(c.strip()) for c in args.ch.split(",") if c.strip().isdigit()
        ]
        if not args.channels:
            raise ValueError
    except ValueError:
        parser.error(
            "❌ Invalid format for --ch. Please enter comma-separated numbers, e.g., 0,2,5,7"
        )

    # Parse cable labels and bind them to Run numbers
    cable_labels = (
        [lbl.strip() for lbl in args.label.split(",") if lbl.strip()]
        if args.label
        else []
    )

    args.run_map = {}
    for i, run_num in enumerate(args.runs):
        if i < len(cable_labels):
            args.run_map[run_num] = cable_labels[i]
        else:
            args.run_map[run_num] = f"Run {run_num}"

    return args


def main():
    args = parse_args()

    # 1. 筛选命令行输入的常规 Run
    normal_plots = []
    for run_num in args.runs:
        cable_name = args.run_map[run_num]
        for afe_num in args.afes:
            for ch in args.channels:
                pattern = os.path.join(
                    DATA_DIR, f"run{run_num}_afe{afe_num}_ch{ch}_*.hdf5"
                )
                matched_files = glob.glob(pattern)
                if matched_files:
                    normal_plots.append((run_num, cable_name, afe_num, ch, matched_files[0]))

    # 2. 筛选硬编码的黑色特殊 Run (1648, 1650, 1652)
    black_plots = []
    for run_num, label_name in BLACK_RUNS_CONFIG.items():
        for afe_num in args.afes:
            for ch in args.channels:
                pattern = os.path.join(
                    DATA_DIR, f"run{run_num}_afe{afe_num}_ch{ch}_*.hdf5"
                )
                matched_files = glob.glob(pattern)
                
                # 兼容不带 AFE/CH 的简版文件名搜索
                if not matched_files:
                    fallback_pattern = os.path.join(DATA_DIR, f"run{run_num}_*.hdf5")
                    matched_files = glob.glob(fallback_pattern)

                if matched_files:
                    if matched_files[0] not in [x[4] for x in black_plots]:
                        black_plots.append((run_num, label_name, afe_num, ch, matched_files[0]))

    total_normal = len(normal_plots)
    total_plots = total_normal + len(black_plots)
    
    if total_plots == 0:
        print(f"🛑 No matching files found in the '{DATA_DIR}' directory!")
        return

    cmap = plt.colormaps.get_cmap("rainbow")

    # 完全恢复你最初的画布尺寸设置
    plt.rcParams["figure.figsize"] = (11, 7)
    plt.figure()

    # 3. 开始循环绘制文件
    # 先画常规 Run（带彩虹颜色渐变）
    for idx, (run_num, cable_name, afe_num, ch, filename) in enumerate(normal_plots):
        display_fname = os.path.basename(filename)
        label_str = f"{cable_name}, AFE {afe_num}, Ch {ch}"

        try:
            with h5py.File(filename, "r") as f:
                wvfms = np.array(f["data"]).astype("float")

            result = mean_fft_analyzer(wvfms)
            display_label = f"{label_str} (RMS: {result.avg_rms:.2f})"
            
            color_ratio = idx / (total_normal - 1) if total_normal > 1 else 0.0

            plt.plot(
                result.x, result.y,
                label=display_label,
                color=cmap(color_ratio),
                alpha=0.8,
                linewidth=1.2,
            )
        except Exception as e:
            print(f"❌ Error processing {display_fname}: {e}")

    # 再画硬编码的黑色 Run
    for idx, (run_num, label_name, afe_num, ch, filename) in enumerate(black_plots):
        display_fname = os.path.basename(filename)
        
        # 保持与你刚才截图完全一致的精简图例命名格式
        if f"ch{ch}" in display_fname.lower():
            label_str = f"{label_name}"
        else:
            label_str = f"{label_name}"

        try:
            with h5py.File(filename, "r") as f:
                wvfms = np.array(f["data"]).astype("float")

            result = mean_fft_analyzer(wvfms)
            display_label = f"{label_str} (RMS: {result.avg_rms:.2f})"

            plt.plot(
                result.x, result.y,
                label=display_label,
                color="black",
                alpha=1.0,
                linewidth=1.5,
                linestyle="-",   # 使用黑色实线
                zorder=100       # 图层强制置顶，不被彩虹色覆盖
            )
        except Exception as e:
            print(f"❌ Error processing {display_fname}: {e}")

    # Plot configuration details (完全恢复原样)
    plt.xscale("log")
    plt.ylim(-120, 0)

    plt.title("DAPHNE Waveform Mean FFT", fontsize=14)
    plt.ylabel("Magnitude [dBFS]", fontsize=12)
    plt.xlabel("Frequency [MHz]", fontsize=12)

    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # 💡 保持放回内部（loc="best"），并自动分为 2 列（ncols=2）横向排列
    # 这样即使有 23 条曲线，图例框也不会向下把图撑爆，完美融入右上角空白处
    plt.legend(
        loc="best",
        ncol=2 if total_plots > 15 else 1,
        frameon=True,
        framealpha=0.8,
        fontsize="small"
    )
    plt.tight_layout()

    print(f"✅ Plotting complete! Successfully plotted {total_plots} curves inside the window.")
    plt.show()


if __name__ == "__main__":
    main()
