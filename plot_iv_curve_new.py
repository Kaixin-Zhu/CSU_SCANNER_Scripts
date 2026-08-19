import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Plot IV Curve with smoothing and derivatives')
parser.add_argument('--filename', type=str, required=True, help='Name of .npz IV curve file inside iv_curve folder')
args = parser.parse_args()

folder = 'iv_curve'
filepath = os.path.join(folder, args.filename)
if not os.path.exists(filepath):
    raise FileNotFoundError(f'File not found: {filepath}')

match = re.search(r'ch\d+', args.filename, re.IGNORECASE)
ch_info = match.group().upper() if match else 'Unknown'

data = np.load(filepath)
current = -1 * data['current']
v = data['v']

# 滑动平均，但保留原长度
window_size = 10
df = pd.DataFrame({'current': current})
current_smooth = df['current'].rolling(window=window_size, min_periods=1, center=True).mean()
current_smooth = np.array(current_smooth)

# 直接用真实的v
logv = np.log(current_smooth)

cs = CubicSpline(v, logv)
deriv_logv = cs.derivative()
cs2 = CubicSpline(v, deriv_logv(v))
deriv2_logv = cs2.derivative()

breakdown_idx = np.argmax(deriv2_logv(v))
breakdown_v = v[breakdown_idx]
breakdown_y = deriv2_logv(v)[breakdown_idx]

plt.plot(v, logv, 'b--', label='ln(I)')
plt.plot(v, deriv_logv(v), 'r-', label="f'ln(I)")
plt.plot(v, deriv2_logv(v), 'g-.', label="f''ln(I)")
plt.plot(breakdown_v, breakdown_y, 'o', color='orange', label=f'Vbd = {breakdown_v:.2f} V')

plt.legend()
plt.xlabel('Bias [V]')
plt.ylabel('current')
plt.title(f'HD module IV curve in Scanner (1st Module, Supercell 1)')
plt.show()
