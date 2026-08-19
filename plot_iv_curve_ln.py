import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

# 解析命令行参数
parser = argparse.ArgumentParser(description='Plot raw ln(I) without smoothing')
parser.add_argument('--filename', type=str, required=True, help='Name of .npz IV curve file inside iv_curve folder')
args = parser.parse_args()

# 文件路径
folder = 'iv_curve'
filepath = os.path.join(folder, args.filename)
if not os.path.exists(filepath):
    raise FileNotFoundError(f'File not found: {filepath}')

# 提取通道信息（如 ch0）
match = re.search(r'ch\d+', args.filename, re.IGNORECASE)
ch_info = match.group().upper() if match else 'Unknown'

# 加载数据
data = np.load(filepath)
v = data['v']
current = -1 * data['current']  # 如果原数据为负电流，则取正

# 不平滑，直接取 ln(I)
log_current = np.log(current)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(v, log_current, 'bo-', label='ln(I)')

plt.xlim(51.2, 51.7)
plt.xlabel("Bias [V]")
plt.ylabel("ln(I)")
plt.title(f"Raw Data (1st module, supercell 1)")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
