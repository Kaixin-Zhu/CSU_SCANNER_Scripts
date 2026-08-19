import argparse
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# ========= 参数 =========
BASELINE_START, BASELINE_END = 100, 200
# ======================

parser = argparse.ArgumentParser(description="DAPHNE 5x4 Channels 2D Histogram Plotter (ROOT-style)")
parser.add_argument(
    "--run",
    type=int,
    required=True,
    help="Specify the run number, e.g. 1719"
)
parser.add_argument(
    "--num_wf",
    type=int,
    default=2000,
    help="Number of waveforms to sample for plotting, default: 2000"
)
args = parser.parse_args()

# 硬件结构定义：5个 AFE Module (5行) x 4个 Channel (4列)
afe_list = [0, 1, 2, 3, 4]    # 5行
ch_list = [0, 2, 5, 7]        # 4列

# 创建 5行 4列 的子图画布
fig, axes = plt.subplots(len(afe_list), len(ch_list), figsize=(18, 11), sharex=True, sharey=True)

print(f"开始绘制 Run {args.run} 的 5x4 2D直方图热力图矩阵...")

# 用于存储最后一个成功绘制的内嵌图像，方便最后生成全局 colorbar
im = None

# 遍历 5行 (AFE Module)
for row_idx, afe in enumerate(afe_list):
    # 遍历 4列 (Channel)
    for col_idx, ch in enumerate(ch_list):
        ax = axes[row_idx, col_idx]
        
        # 匹配文件
        pattern = f"data/run{args.run}_afe{afe}_ch{ch}_*.hdf5"
        matched_files = glob(pattern)
        
        if not matched_files:
            pattern_alt = f"run{args.run}_afe{afe}_ch{ch}.hdf5"
            matched_files = glob(pattern_alt)
            
        if not matched_files:
            ax.text(0.5, 0.5, f"No File\nAFE{afe}_CH{ch}",
                    color="gray", ha="center", va="center", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            continue
            
        filename = matched_files[0]
        
        try:
            with h5py.File(filename, 'r') as f:
                wvfms = np.array(f['data'], dtype=np.float64)
            
            total_wvfms = wvfms.shape[0]
            
            # ===== 信号翻转 =====
            wvfms = -wvfms
            
            # ===== 逐行独立进行基线校准 =====
            baselines = wvfms[:, BASELINE_START:BASELINE_END].mean(axis=1, keepdims=True)
            adjusted_wvfms = wvfms - baselines
            
            # 均匀抽样波形
            step = max(1, total_wvfms // args.num_wf)
            sampled_wvfms = adjusted_wvfms[::step]
            
            # ===== 构建 2D 直方图的数据输入 =====
            num_samples = sampled_wvfms.shape[1]
            
            # 构造与数据对齐的 X 坐标阵列 (每个点对应其时间/采样点 index)
            x_data = np.tile(np.arange(num_samples), (sampled_wvfms.shape[0], 1)).flatten()
            y_data = sampled_wvfms.flatten()
            
            # ===== 设置直方图网格 (Bins) =====
            # X轴按采样点 1:1 分 bin；Y轴在 -8000 到 8000 间分成 200 个 bin（可根据需要微调）
            x_bins = np.arange(0, num_samples + 1, 1)
            y_bins = np.linspace(-8000, 8000, 201)
            
            # ===== 绘制 ROOT 风格的 2D 直方图 =====
            # cmap 可以用 'jet'（最像 ROOT 默认）、'viridis'（现代、感知均匀）或 'plasma'
            # norm=colors.LogNorm() 开启对数刻度，能更清晰地看清边缘低频噪声和核心强信号
            im = ax.hist2d(
                x_data, y_data,
                bins=[x_bins, y_bins],
                cmap='jet',
                norm=colors.LogNorm(vmin=1, vmax=args.num_wf//4),
                cmin=1 # 计数为0的格子不染色（保持白底/透明）
            )[3]
            
            # 依然保留平均值虚线作为参考
            avg_wvfm = adjusted_wvfms.mean(axis=0)
            ax.plot(avg_wvfm, lw=0.8, color="black", alpha=0.6, linestyle=":")
            
            ax.set_ylim(-8000, 8000)
            ax.grid(True, linestyle='--', alpha=0.2)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\nAFE{afe}_CH{ch}",
                    color="crimson", ha="center", va="center", fontsize=10)
            continue
            
        # --- 干净的坐标轴标签逻辑 ---
        if col_idx == 0:
            ax.set_ylabel(f"AFE {afe}", fontsize=12)
            
        if row_idx == len(afe_list) - 1:
            ax.set_xlabel(f"CH {ch}", fontsize=12)

# 全局大标题
fig.suptitle(f"Scanner Waveform Persistence - Run {args.run}",
             fontsize=16)

# 在右侧腾出空间放置全局的 Colorbar
fig.tight_layout(rect=[0.01, 0.02, 0.90, 0.95])

if im is not None:
    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.10, 0.02, 0.78]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Hits / Bin Count', fontsize=12, fontweight='bold')

print("5x4 ROOT风格2D直方图矩阵生成成功！")
plt.show()
