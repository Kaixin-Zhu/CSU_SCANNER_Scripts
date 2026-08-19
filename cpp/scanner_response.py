import argparse
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# ==========================================================
# ### CONFIGURATION (基础配置) ###
# ==========================================================
BASELINE_START, BASELINE_END = 100, 200
THRESHOLD = 25.0  # 临界判定线 25%

# ⭐ 使用你代码中完全一致的 4 个固定 pulse 窗口
WINDOWS = [(600, 900), (1500, 1900), (2400, 2800), (3300, 3700)]
# ==========================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Module Response 3: Symmetric Absolute Normalized Heatmap (Pure Red Edition)"
    )
    parser.add_argument("--run", type=int, required=True, help="Test run number (e.g., 1722)")
    parser.add_argument("--ref", type=int, required=True, help="Reference run number (e.g., 1720)")
    parser.add_argument("--afes", type=str, default="4,3,2,1,0", help="AFEs to analyze, default: 4,3,2,1,0")
    parser.add_argument("--chs", type=str, default="0,2,5,7", help="Channels, default: 0,2,5,7")

    args = parser.parse_args()
    args.afe_list = [int(a.strip()) for a in args.afes.split(",") if a.strip().isdigit()]
    args.ch_list = sorted([int(c.strip()) for c in args.chs.split(",") if c.strip().isdigit()])
    return args


def extract_afe_ch_signal(run, afe, ch):
    """提取单个 run, afe, ch 对应的 4 个 pulse 振幅"""
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

    avg_wvfm = -avg_wvfm
    baseline = avg_wvfm[BASELINE_START:BASELINE_END].mean()
    adjusted = avg_wvfm - baseline

    amps = []
    for start, end in WINDOWS:
        segment = adjusted[start:end]
        idx_local = np.argmax(segment)
        amp = segment[idx_local]
        amps.append(float(amp))
        
    return amps


def main():
    args = parse_args()

    rows_num = len(args.afe_list)
    cols_num = len(args.ch_list) * 4
    deviation_matrix = np.full((rows_num, cols_num), np.nan)

    print(f"🔄 正在计算 Run {args.run} 相对 Run {args.ref} 的归一化偏差...")

    for row_idx, afe in enumerate(args.afe_list):
        for ch_idx, ch in enumerate(args.ch_list):
            amps_test = extract_afe_ch_signal(args.run, afe, ch)
            amps_ref = extract_afe_ch_signal(args.ref, afe, ch)
            
            if (amps_test is not None) and (amps_ref is not None):
                amps_test = np.array(amps_test)
                amps_ref = np.array(amps_ref)
                deviation_pct = (amps_test / amps_ref - 1.0) * 100.0
                
                start_col = ch_idx * 4
                end_col = start_col + 4
                deviation_matrix[row_idx, start_col:end_col] = deviation_pct

    # ==========================================================
    # ### 绘图部分（对称绝对值：大红警示版） ###
    # ==========================================================
    fig, axes = plt.subplots(rows_num, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={'hspace': 0.3})
    
    if rows_num == 1:
        axes = [axes]

    # ⭐ 核心修改：将两端的颜色代码由 "#d62728"（深褐红）替换为 "#FF0000"（纯正大红）
    cdict = [
        (0.0, "#FF0000"),   # -25% 是鲜艳大红 (Pure Red)
        (0.32, "#ff7f0e"),  # -9%  左右是警示橙色 (Orange)
        (0.5, "#2ca02c"),   #  0%  完美中心点：你的科研绿 (Green)
        (0.68, "#ff7f0e"),  # +9%  左右是警示橙色 (Orange)
        (1.0, "#FF0000")    # +25% 是鲜艳大红 (Pure Red)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("Symmetric_Green_To_PureRed", cdict, N=256)
    norm = mcolors.Normalize(vmin=-THRESHOLD, vmax=THRESHOLD)

    # 循环绘制每一个 Module (行)
    for r, afe in enumerate(args.afe_list):
        ax = axes[r]
        row_data = np.atleast_2d(deviation_matrix[r, :])
        
        im = ax.imshow(row_data, cmap=cmap, norm=norm, aspect='auto',
                       extent=[-0.5, cols_num - 0.5, -0.5, 0.5])
        
        ax.set_yticks([])
        ax.set_ylabel(f"AFE {afe}", fontsize=11, rotation=0, ha='right', va='center')
        
        # 填充文本数值（统一使用白色粗体）
        for c in range(cols_num):
            val = deviation_matrix[r, c]
            if not np.isnan(val):
                text_str = f"{val:+.1f}%"
                val_abs = abs(val)
                
                if val_abs > THRESHOLD:
                    text_color = "yellow"  # 极少数越界到 25% 以上的用黄色高亮
                    font_weight = "bold"
                else:
                    text_color = "white"   # 统一白色粗体
                    font_weight = "bold"
                    
                ax.text(c, 0, text_str, color=text_color, ha="center", va="center", fontsize=9, fontweight=font_weight)
            else:
                ax.text(c, 0, "NaN", color="gray", ha="center", va="center", fontsize=9)

        # 绘制各种物理边界分割线
        ax.axhline(-0.5, color='gray', linewidth=0.5)
        ax.axhline(0.5, color='gray', linewidth=0.5)
        for c in range(cols_num + 1):
            if c % 4 == 0:
                ax.axvline(c - 0.5, color='black', linewidth=2.5)  # Supercell 粗黑边界线
            else:
                ax.axvline(c - 0.5, color='gray', linewidth=0.5, linestyle='--')

    # 底部主坐标轴标签
    last_ax = axes[-1]
    last_ax.set_xticks(np.arange(cols_num))
    last_ax.set_xticklabels([f"P{i%4}" for i in range(cols_num)], fontsize=9)

    # 底部 Supercell 通道归属标签
    for ch_idx, ch in enumerate(args.ch_list):
        last_ax.text(ch_idx * 4 + 1.5, -0.9, f"CH {ch}",
                     color='black', fontsize=12, ha='center', va='top')

    # 最右侧专用色条区域 (防止覆盖通道方块)
    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.08, right=0.84)
    cbar_ax = fig.add_axes([0.88, 0.25, 0.015, 0.55])  # [左, 下, 宽, 高]
    
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both')
    cbar.set_label('Normalization Deviation (%)', fontsize=12, labelpad=10)
    cbar.set_ticks([-THRESHOLD, -15, 0, 15, THRESHOLD])
    cbar.ax.set_yticklabels([f'-{THRESHOLD}% (Red)', '-15% (Orange)', '0% (Green)', '+15% (Orange)', f'+{THRESHOLD}% (Red)'])

    # 全局大标题
    fig.suptitle(f"Module Response 3 (Normalized Comparison)\n \n Run {args.run} vs Reference Run {args.ref}",
                 fontsize=14)

    print("📊 大红改良版色块图生成完毕，再跑一次看看效果！")
    plt.show()

if __name__ == "__main__":
    main()
