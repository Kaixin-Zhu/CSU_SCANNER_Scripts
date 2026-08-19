import argparse
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# ========= 参数 =========
BASELINE_START, BASELINE_END = 100, 200
# ======================

parser = argparse.ArgumentParser(description="Scanner Averaged Waveform")
parser.add_argument(
    "--run",
    type=int,
    required=True,
    help="Specify the run number, e.g. 1719"
)
args = parser.parse_args()

# ⭐ 按照硬件结构定义：5个 AFE Module (5行) x 4个 Channel (4列)
afe_list = [0,1,2,3,4]    # 5行
ch_list = [0, 2, 5, 7]        # 4列

# 创建 5行 4列 的子图画布，共享X轴和Y轴刻度使对比更直观
fig, axes = plt.subplots(len(afe_list), len(ch_list), figsize=(16, 10), sharex=True, sharey=True)

print(f"开始绘制 Run {args.run} 的 5x4 AFE-CH 波形矩阵...")

# 遍历 5行 (AFE Module)
for row_idx, afe in enumerate(afe_list):
    # 遍历 4列 (Channel)
    for col_idx, ch in enumerate(ch_list):
        ax = axes[row_idx, col_idx]
        
        # 根据你数据目录的典型命名规则匹配文件
        pattern = f"data/run{args.run}_afe{afe}_ch{ch}_*.hdf5"
        matched_files = glob(pattern)
        
        if not matched_files:
            # 兼容处理：如果没有 data/ 目录，尝试直接匹配当前目录下的简化命名文件
            pattern_alt = f"run{args.run}_afe{afe}_ch{ch}.hdf5"
            matched_files = glob(pattern_alt)
            
        if not matched_files:
            # 文件不存在时留白提示
            ax.text(0.5, 0.5, f"No File\nAFE{afe}_CH{ch}",
                    color="gray", ha="center", va="center", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            continue
            
        filename = matched_files[0]
        
        try:
            with h5py.File(filename, 'r') as f:
                wvfms = np.array(f['data'], dtype=np.float64)
                
            # 计算平均波形 (axis=0 代表对10000个波形按点求均值)
            avg_wvfm = np.mean(wvfms, axis=0)
            
            # ===== 信号翻转与基线校准 =====
            avg_wvfm = -avg_wvfm  # 翻转负脉冲
            baseline = np.mean(avg_wvfm[BASELINE_START:BASELINE_END])
            adjusted_avg = avg_wvfm - baseline
            
            # ===== 绘制波形 =====
            # 每一行（同一个 AFE）用同一个独特的颜色，方便横向对比
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            ax.plot(adjusted_avg, lw=1.5, color=colors[row_idx], label=f"M{afe}_CH{ch}")
            
            ax.grid(True, linestyle='--', alpha=0.5)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\nAFE{afe}_CH{ch}",
                    color="crimson", ha="center", va="center", fontsize=10)
            continue
            
        # --- 干净的坐标轴标签逻辑 ---
        # 只在第一列（最左边）标出 AFE Module 行名
        if col_idx == 0:
            ax.set_ylabel(f"AFE {afe}", fontsize=12)
            
        # 只在最后一行（最底下）标出 Channel 列名
        if row_idx == len(afe_list) - 1:
            ax.set_xlabel(f"CH {ch}", fontsize=12)

# 全局大标题与紧凑布局
fig.suptitle(f"Scanner Averaged Waveform - Run {args.run}",
             fontsize=16)

plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
print("5x4 画布生成成功！")
plt.show()
