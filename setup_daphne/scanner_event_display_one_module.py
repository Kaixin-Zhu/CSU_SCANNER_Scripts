#!/usr/bin/env python3
############################################################
# Live-updating event display for DAPHNE using spy buffers
# Author: Sam Fogarty, modified by Kaixin Zhu
# Supports 4 channels in 2x2 grid with peak-side amplitude display
# Added: Argument parser for dynamically selecting AFE modules
############################################################

import ROOT
import time
from array import array
import numpy as np
import DaphneInterface as ivtools
import sys
import argparse

def signal_handler(sig, frame):
    print('\nCtrl+C detected. Exiting gracefully.')
    sys.exit(0)

def calculate_amplitudes_with_pos(y_data):
    """
    计算单条波形中的 4 个负脉冲幅度，并返回它们在图表中的确切 (X, Y) 物理坐标
    """
    # 1. 用前 200 个点计算当前的动态 baseline
    baseline = np.mean(y_data[:200])
    
    # 2. 根据脉冲位置划分 4 个时间窗
    windows = [
        (500, 1100),   # Peak 1
        (1400, 2000),  # Peak 2
        (2300, 2900),  # Peak 3
        (3300, 3900)   # Peak 4
    ]
    
    amp_details = []
    for start, end in windows:
        # 找到区间内最低点的相对索引
        local_min_idx = np.argmin(y_data[start:end])
        # 换算成绝对 Sample 坐标 (X 轴)
        peak_x = start + local_min_idx
        # 拿到最低点的 ADC 原始值 (Y 轴)
        peak_y = y_data[peak_x]
        
        # 负信号幅度 = 基线 - 最低点
        amp = baseline - peak_y
        
        amp_details.append({
            'amp': int(amp),
            'x': peak_x,
            'y': peak_y
        })
        
    return amp_details

def update_canvas(canvas, x_data, y_avg_list, channels, afe_num):
    """
    更新 2x2 画布，并将幅度值直接动态标注在每个负脉冲波谷的下方
    """
    for idx, ch in enumerate(channels):
        # 切换到对应的子区域 (1 到 4)
        canvas.cd(idx + 1)
        ROOT.gPad.Clear()

        y_data = y_avg_list[idx]
        amp_details = calculate_amplitudes_with_pos(y_data)
        
        # 转换数据为 ROOT 格式并绘图
        graph = ROOT.TGraph(len(x_data), array('d', x_data), array('d', y_data))
        colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta+2]
        current_color = colors[idx % len(colors)]
        
        graph.SetLineColor(current_color)
        # 标题中加入 AFE 信息，方便确认当前看的是哪一块
        graph.SetTitle(f'AFE {afe_num} - Channel {ch} ;Tick;ADC')

        # 动态设置 Y 轴范围：为了防止底部的文字越界，下方留出 25% 的 margin
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        y_diff = y_max - y_min if y_max != y_min else 10
        
        graph.GetYaxis().SetRangeUser(y_min - 0.25 * y_diff, y_max + 0.1 * y_diff)

        # 显式接管生命周期，防止 TGraph 闪烁或消失
        ROOT.SetOwnership(graph, False)
        graph.Draw("ALP")

        # ================= 动态在波谷旁边绘制幅度 =================
        for pt in amp_details:
            latex = ROOT.TLatex()
            latex.SetTextSize(0.04)
            latex.SetTextColor(current_color)  # 字体颜色和波形保持一致
            latex.SetTextAlign(23)            # 水平居中，垂直靠顶对齐

            # 文字的放置坐标：X 轴对齐波谷，Y 轴位于波谷下方 400 个 ADC 计数处
            text_x = pt['x']
            text_y = pt['y'] - 400
            
            latex.DrawLatex(text_x, text_y, f"{pt['amp']}")
            ROOT.SetOwnership(latex, False)
        # =========================================================

    canvas.Update()
    ROOT.gSystem.ProcessEvents()


def main():
    # ================= Command Line Arguments =================
    parser = argparse.ArgumentParser(description="Live-updating event display for a selected DAPHNE AFE module.")
    parser.add_argument('--afe', type=int, required=True, choices=[0, 1, 2, 3, 4], help="AFE module index to monitor (0-4)")
    args = parser.parse_args()
    
    afe = args.afe

    # ================= Configuration =================
    length = 4000
    chunk_length = 50
    chunks = int(length / chunk_length)

    base_register = 0x40000000
    AFE_hex_base = 0x100000
    Channel_hex_base = 0x10000

    ip = '10.73.137.110'
    device = ivtools.daphne(ip)
    print("DAPHNE firmware version %0X" % device.read_reg(0x9000, 1)[2])
    print(f"Successfully connected. Now monitoring AFE {afe}...")

    channels = [0, 2, 5, 7]  # 4个监控通道
    num_ch = len(channels)
    nWvfms_avg = 10
    do_software_trigger = False

    # ================= Setup Canvas =================
    canvas = ROOT.TCanvas("canvas", f"Scanner Monitor - AFE {afe}", 1000, 800)
    canvas.Divide(2, 2)

    last_timestamp = -1
    x_data = np.arange(0, length)

    # ================= Data Containers =================
    y_data_ch = [np.zeros(length) for _ in range(num_ch)]
    y_data_avg = [np.zeros(length) for _ in range(num_ch)]

    y_data_list_last = [np.zeros((nWvfms_avg, length)) for _ in range(num_ch)]
    y_data_list_next = [np.zeros((nWvfms_avg, length)) for _ in range(num_ch)]

    nWvfms_abs = 1
    nWvfms = 1

    # ================= Main Loop =================
    try:
        while True:
            if nWvfms_avg == 1:
                time.sleep(0.1)
            if do_software_trigger:
                device.write_reg(0x2000, [1234])

            current_timestamp = int(device.read_reg(0x40500000, 1)[2])
            if last_timestamp != current_timestamp:
                last_timestamp = current_timestamp

                # 读取 4 个通道的数据
                for idx, ch in enumerate(channels):
                    for i in range(chunks):
                        reg_addr = base_register + AFE_hex_base * afe + Channel_hex_base * ch + i * chunk_length
                        doutrec = device.read_reg(reg_addr, chunk_length)
                        y_data_ch[idx][i * chunk_length:(i + 1) * chunk_length] = doutrec[2:]

                # 滚动平均处理
                for idx in range(num_ch):
                    if nWvfms_abs < nWvfms_avg:
                        y_data_list_last[idx][nWvfms - 1, :] = y_data_ch[idx]
                    elif nWvfms_abs == nWvfms_avg:
                        y_data_list_last[idx][nWvfms - 1, :] = y_data_ch[idx]
                        y_data_avg[idx] = np.sum(y_data_list_last[idx], axis=0) / nWvfms_avg
                    else:
                        y_data_list_next[idx][nWvfms - 1, :] = y_data_ch[idx]
                        y_data_avg[idx] = (
                            np.sum(y_data_list_last[idx][nWvfms:], axis=0) +
                            np.sum(y_data_list_next[idx][0:nWvfms], axis=0)
                        ) / nWvfms_avg

                # 刷新画布
                if nWvfms_abs >= nWvfms_avg:
                    update_canvas(canvas, x_data, y_data_avg, channels, afe)

                # 滚动平均的缓存替换
                if nWvfms == nWvfms_avg and nWvfms_abs >= 2 * nWvfms_avg:
                    for idx in range(num_ch):
                        y_data_list_last[idx] = np.copy(y_data_list_next[idx])

                nWvfms = 1 if nWvfms == nWvfms_avg else nWvfms + 1
                nWvfms_abs += 1

            time.sleep(0.02)

    except KeyboardInterrupt:
        print('\nCtrl+C detected. Exiting gracefully.')
        sys.exit(0)

if __name__ == "__main__":
    main()
