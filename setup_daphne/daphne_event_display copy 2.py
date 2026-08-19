############################################################
# Live-updating event display for DAPHNE using spy buffers
# Author: Sam Fogarty, fogar314@colostate.edu
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

def update_canvas(canvas, x_data, y_data):
    canvas.Clear()
    graph = ROOT.TGraph(len(x_data), array('d', x_data), array('d', y_data))
    graph.SetLineColor(ROOT.kBlue)
    graph.Draw("ALP")
    graph.SetTitle('LED-Generated Photon Detector Signals')
    canvas.Update()
    ROOT.gSystem.ProcessEvents()

# Configuration
length = 4000
chunk_length = 50
chunks = int(length / chunk_length)

base_register = 0x40000000
AFE_hex_base = 0x100000
Channel_hex_base = 0x10000

ip = '10.73.137.110'
device = ivtools.daphne(ip)
print("DAPHNE firmware version %0X" % device.read_reg(0x9000, 1)[2])

afe, chan = 0, 2
nWvfms_avg = 10
do_software_trigger = False

# Setup
canvas = ROOT.TCanvas("canvas", "Dynamic Canvas", 800, 600)

last_timestamp = -1
x_data = np.arange(0, length)
y_data = np.zeros(length)
y_data_avg = np.zeros(length)

y_data_list_last = np.zeros((nWvfms_avg, length))
y_data_list_next = np.zeros((nWvfms_avg, length))
nWvfms_abs = 1
nWvfms = 1
both_lists_filled = False

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

            # Read waveform
            for i in range(chunks):
                doutrec = device.read_reg(
                    base_register + AFE_hex_base * afe + Channel_hex_base * chan + i * chunk_length,
                    chunk_length
                )
                y_data[i * chunk_length:(i + 1) * chunk_length] = doutrec[2:]

            # Rolling average logic
            print(f"{nWvfms_abs=}, {nWvfms=}")

            if nWvfms_abs < nWvfms_avg:
                y_data_list_last[nWvfms - 1, :] = y_data
            elif nWvfms_abs == nWvfms_avg:
                y_data_list_last[nWvfms - 1, :] = y_data
                y_data_avg = np.sum(y_data_list_last, axis=0) / nWvfms_avg
                print("Updating canvas")
                update_canvas(canvas, x_data, y_data_avg)
            else:
                y_data_list_next[nWvfms - 1, :] = y_data
                y_data_avg = (
                    np.sum(y_data_list_last[nWvfms:], axis=0) +
                    np.sum(y_data_list_next[0:nWvfms], axis=0)
                ) / nWvfms_avg
                print("Updating canvas")
                update_canvas(canvas, x_data, y_data_avg)

            if nWvfms == nWvfms_avg and nWvfms_abs >= 2 * nWvfms_avg:
                y_data_list_last = np.copy(y_data_list_next)

            nWvfms = 1 if nWvfms == nWvfms_avg else nWvfms + 1
            nWvfms_abs += 1

        time.sleep(0.02)

except KeyboardInterrupt:
    print('Ctrl+C detected. Exiting gracefully.')
    sys.exit(0)
