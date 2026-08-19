# set_bias.py Converts desired bias voltage to DAC values and sends it to appropriate AFE
# Uses the new SPI slave firmware. Python 3.

import DaphneInterface as ivtools
import argparse
import numpy as np
from tqdm import tqdm
import os

def main(ep, afe, ch, run_type, run_id):
    afe = int(afe)
    ch = int(ch)

    device = ivtools.daphne(f"10.73.137.{ep}")

    def DAC_for_V(afe, v):
        if afe == 0:
            return abs(round((v - 0.053) / 0.0394))
        elif afe == 1:
            return abs(round((v - 0.0447) / 0.0391))
        elif afe == 2:
            return abs(round((v - 0.00945) / 0.0392))
        elif afe == 3:
            return abs(round((v + 0.371) / 0.0391))
        elif afe == 4:
            return abs(round((v - 0.00328) / 0.0391))
        else:
            raise ValueError("We only have AFE's 0-4!")

    if run_type == 'warm':
        vlimit = 53
        vrange = [50.5, 52.9]
    elif run_type == 'cold':
        vlimit = 47
        vrange = [40, 47]
    else:
        raise Exception('run_type not recognized. warm or cold are only options')

    print('WR VBIASCTRL V 4095')
    response_data = device.command('WR VBIASCTRL V 4095')

    viter = 0.04
    v_list = np.arange(vrange[0], vrange[1] + viter, viter)
    v_list = v_list[v_list <= vrange[1]]
    current = []

    for v in v_list:
        print(f'v = {v}')
        if v == 0:
            dac = 0
        elif v > vlimit:
            raise Exception(f'Cannot set bias higher than {vlimit}')
        else:
            dac = DAC_for_V(afe, v)

        CmdString = f"WR AFE {afe} BIASSET V {dac}"
        device.command(CmdString)
        I = device.read_current(8 * afe + ch, 15)
        current.append(I)
        print(f'current = {I}')

    save_dir = "iv_curve"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"run{run_id}_ch{ch}_{run_type}.npz"
    save_path = os.path.join(save_dir, filename)
    np.savez(save_path, current=current, v=v_list)
    print(f"\n IV curve save to {save_path}")

    device.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAPHNE command interpreter')
    parser.add_argument('--ep', required=True, help='DAPHNE IP endpoint, e.g. 110')
    parser.add_argument('--afe', required=True, help='Which AFE are we setting bias for? Number 0-4')
    parser.add_argument('--ch', required=True, help='Which channel to check current? Number 0-7')
    parser.add_argument('--run_type', required=True, help='Run type. Choose warm for warm test and cold for cold test.')
    parser.add_argument('--run', required=True, type=int, help='Run number')
    args = parser.parse_args()

    main(args.ep, args.afe, args.ch, args.run_type, args.run)

