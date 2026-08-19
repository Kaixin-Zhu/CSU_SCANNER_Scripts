import argparse
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# ### CONFIGURATION (基础配置) ###
# ==========================================================
BASELINE_START, BASELINE_END = 100, 200

# ⭐ 固定4个pulse窗口
WINDOWS = [(600, 900), (1500, 1900), (2400, 2800), (3300, 3700)]
# ==========================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DAPHNE LED Response & Peak Distribution Analyzer"
    )

    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Run numbers, separated by commas (e.g., 1720)",
    )
    # 默认遍历这 5 个 AFE
    parser.add_argument(
        "--afes",
        type=str,
        default="4,3,2,1,0",
        help="AFEs to analyze, default: 4,3,2,1,0",
    )
    # 默认 4 个通道
    parser.add_argument(
        "--chs",
        type=str,
        default="0,2,5,7",
        help="Channels, separated by commas, default: 0,2,5,7",
    )

    args = parser.parse_args()

    try:
        args.run_list = [int(r.strip()) for r in args.runs.split(",") if r.strip().isdigit()]
        if not args.run_list: raise ValueError
    except ValueError:
        parser.error("❌ --runs 参数格式错误")

    try:
        args.afe_list = [int(a.strip()) for a in args.afes.split(",") if a.strip().isdigit()]
    except ValueError:
        parser.error("❌ --afes 参数格式错误")

    try:
        args.ch_list = [int(c.strip()) for c in args.chs.split(",") if c.strip().isdigit()]
        if not args.ch_list: raise ValueError
    except ValueError:
        parser.error("❌ --chs 参数格式错误")

    return args


def extract_afe_ch_signal(run, afe, ch):
    """提取单个 run, 单个 afe, 单个 ch 的4个 pulse 振幅"""
    pattern = f"data/run{run}_afe{afe}_ch{ch}_*.hdf5"
    matched_files = glob(pattern)

    if not matched_files:
        return None

    filename = matched_files[0]
    try:
        with h5py.File(filename, "r") as f:
            wvfms = np.array(f["data"], dtype=np.float64)
            avg_wvfm = wvfms.mean(axis=0)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return None

    # ===== baseline =====
    avg_wvfm = -avg_wvfm
    baseline = avg_wvfm[BASELINE_START:BASELINE_END].mean()
    adjusted = avg_wvfm - baseline

    # ===== 固定窗口找4个pulse =====
    amps = []
    for start, end in WINDOWS:
        segment = adjusted[start:end]
        idx_local = np.argmax(segment)
        amp = segment[idx_local]
        amps.append(float(amp))
        
    return amps


def main():
    args = parse_args()

    # 既然有 5 条线，用比较经典的 tab10 调色盘分色会更明显、更好看
    cmap = plt.colormaps.get_cmap("tab10")

    # ====== Plot ======
    plt.figure(figsize=(10, 6))
    
    all_runs_signals = []  # 用于计算全局最大差值
    plot_data = []         # 缓存每条线的数据

    # 核心修改：让 AFE 独立成为一组线
    line_idx = 0
    for run in args.run_list:
        for afe in args.afe_list:
            afe_signals = []
            
            # 按顺序把当前 AFE 的所有 ch 数据拼起来
            for ch in sorted(args.ch_list):
                amps = extract_afe_ch_signal(run, afe, ch)
                if amps is not None:
                    afe_signals.extend(amps)
            
            # 如果这个 AFE 成功收集到了数据，加入绘图缓存
            if len(afe_signals) > 0:
                all_runs_signals.extend(afe_signals)
                plot_data.append((run, afe, afe_signals, line_idx))
                line_idx += 1

    if not all_runs_signals:
        print("🛑 data/ 文件夹下没有找到任何可匹配处理的 AFE/CH 文件！")
        return

    # 计算全局最大差值
    global_max = np.max(all_runs_signals)
    global_min = np.min(all_runs_signals)
    global_diff = global_max - global_min

    handles, labels = [], []

    # 开始循环绘制 5 条 AFE 曲线
    for run, afe, signals, idx in plot_data:
        max_peak = np.max(signals)
        min_peak = np.min(signals)
        peak_diff = max_peak - min_peak

        x = np.arange(1, len(signals) + 1)
        label_str = f"Run {run} AFE {afe} (Diff: {peak_diff:.1f})"

        (line,) = plt.plot(
            x,
            signals,
            marker="o",
            linewidth=1.5,
            color=cmap(idx % 10), # 循环分配不同颜色
            label=label_str,
        )
        handles.append(line)
        labels.append(label_str)

    # ====== 追加 Global Max Diff 到 Legend ======
    blank_handle = plt.plot([], [], linestyle="None", marker="None", label="")[0]
    handles.append(blank_handle)
    labels.append(f"Global Max Diff: {global_diff:.1f}")

    # ====== 美化 ======
    plt.title("LED Response Across Scanner", fontsize=14, fontweight="bold")
    plt.xlabel("LED Index", fontsize=12)
    plt.ylabel("Amplitude (ADC)", fontsize=12)
    plt.grid(alpha=0.3, linestyle="--")
    plt.ylim(0, 8000)

    # 根据实际输入的 channel 数量动态绘制分割线
    for i in range(1, len(args.ch_list)):
        plt.axvline(i * 4 + 0.5, linestyle="--", color="gray", alpha=0.4)

    # 图例位置移入左下角
    plt.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        borderaxespad=1.2,
        fontsize="small",
        framealpha=0.9
    )
    
    plt.tight_layout()

    # SC中心位置标记
    ax = plt.gca()
    sc_centers = [2, 6, 10, 14]
    sc_labels = ["SC1", "SC2", "SC3", "SC4"]
    y_top = ax.get_ylim()[1]

    for xc, label in zip(sc_centers, sc_labels):
        ax.text(
            xc,
            y_top * 0.95,
            label,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            alpha=0.7
        )

    print(f"✅ 成功绘制！共生成 {len(plot_data)} 条 AFE 数据曲线。")
    plt.show()


if __name__ == "__main__":
    main()
