############################################################
# Live-updating event display for DAPHNE using spy buffers
# Author: Sam Fogarty, modified by Kaixin Zhu
# Now supports displaying two channels simultaneously
############################################################

import ROOT
import time
from array import array
import numpy as np
import DaphneInterface as ivtools
import sys

def signal_handler(sig, frame):
    print('Ctrl+C detected. Exiting gracefully.')
    sys.exit(0)

def update_canvas(canvas, x_data, y1_data, y2_data):
    canvas.Clear()

    graph1 = ROOT.TGraph(len(x_data), array('d', x_data), array('d', y1_data))
    graph1.SetLineColor(ROOT.kBlue)
    graph1.SetTitle('LED-Generated Photon Detector Signals;Sample;ADC counts')

    graph2 = ROOT.TGraph(len(x_data), array('d', x_data), array('d', y2_data))
    graph2.SetLineColor(ROOT.kRed)

    # ================== 核心：统一 Y 轴范围 ==================
    y_max = max(np.max(y1_data), np.max(y2_data))
    y_min = min(np.min(y1_data), np.min(y2_data))

    margin = 0.1 * (y_max - y_min) if y_max != y_min else 10
    graph1.GetYaxis().SetRangeUser(y_min - margin, y_max + margin)
    # ========================================================

    graph1.Draw("ALP")
    graph2.Draw("LP SAME")

    legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    legend.AddEntry(graph1, "Channel 5", "l")
    legend.AddEntry(graph2, "Channel 7", "l")
    legend.Draw()

    canvas.Update()
    ROOT.gSystem.ProcessEvents()


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

afe = 0
channels = [5,7]  # two channels to display
nWvfms_avg = 10
do_software_trigger = False

# ================= Setup =================
canvas = ROOT.TCanvas("canvas", "Dynamic Canvas", 800, 600)

last_timestamp = -1
x_data = np.arange(0, length)
y_data_ch1 = np.zeros(length)
y_data_ch2 = np.zeros(length)
y_data_avg1 = np.zeros(length)
y_data_avg2 = np.zeros(length)

y_data_list_last1 = np.zeros((nWvfms_avg, length))
y_data_list_next1 = np.zeros((nWvfms_avg, length))
y_data_list_last2 = np.zeros((nWvfms_avg, length))
y_data_list_next2 = np.zeros((nWvfms_avg, length))

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
            print('New trigger detected!')
            last_timestamp = current_timestamp

            # Read waveform for both channels
            for i in range(chunks):
                # Channel 5
                doutrec_5 = device.read_reg(
                    base_register + AFE_hex_base * afe + Channel_hex_base * channels[0] + i * chunk_length,
                    chunk_length
                )
                y_data_ch1[i * chunk_length:(i + 1) * chunk_length] = doutrec_5[2:]

                # Channel 7
                doutrec_7 = device.read_reg(
                    base_register + AFE_hex_base * afe + Channel_hex_base * channels[1] + i * chunk_length,
                    chunk_length
                )
                y_data_ch2[i * chunk_length:(i + 1) * chunk_length] = doutrec_7[2:]

            # Rolling average logic
            if nWvfms_abs < nWvfms_avg:
                y_data_list_last1[nWvfms - 1, :] = y_data_ch1
                y_data_list_last2[nWvfms - 1, :] = y_data_ch2
            elif nWvfms_abs == nWvfms_avg:
                y_data_list_last1[nWvfms - 1, :] = y_data_ch1
                y_data_list_last2[nWvfms - 1, :] = y_data_ch2
                y_data_avg1 = np.sum(y_data_list_last1, axis=0) / nWvfms_avg
                y_data_avg2 = np.sum(y_data_list_last2, axis=0) / nWvfms_avg
                print("Updating canvas")
                update_canvas(canvas, x_data, y_data_avg1, y_data_avg2)
            else:
                y_data_list_next1[nWvfms - 1, :] = y_data_ch1
                y_data_list_next2[nWvfms - 1, :] = y_data_ch2
                y_data_avg1 = (
                    np.sum(y_data_list_last1[nWvfms:], axis=0) +
                    np.sum(y_data_list_next1[0:nWvfms], axis=0)
                ) / nWvfms_avg
                y_data_avg2 = (
                    np.sum(y_data_list_last2[nWvfms:], axis=0) +
                    np.sum(y_data_list_next2[0:nWvfms], axis=0)
                ) / nWvfms_avg
                print("Updating canvas")
                update_canvas(canvas, x_data, y_data_avg1, y_data_avg2)

            if nWvfms == nWvfms_avg and nWvfms_abs >= 2 * nWvfms_avg:
                y_data_list_last1 = np.copy(y_data_list_next1)
                y_data_list_last2 = np.copy(y_data_list_next2)

            nWvfms = 1 if nWvfms == nWvfms_avg else nWvfms + 1
            nWvfms_abs += 1

        time.sleep(0.02)

except KeyboardInterrupt:
    print('Ctrl+C detected. Exiting gracefully.')
    sys.exit(0)
