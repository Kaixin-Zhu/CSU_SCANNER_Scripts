# set_bias.py Converts desired bias voltage to DAC values and sends it to appropriate AFE
# Uses the new SPI slave firmware. Python 3.

import argparse
import DaphneInterface as ivtools


def main(ep, afe_list, v, run_type):

    v = float(v)
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
            raise ValueError(f"We only have AFE's 0-4! Invalid AFE: {afe}")

    if run_type == "warm":
        vlimit = 55
    elif run_type == "cold":
        vlimit = 47
    else:
        raise Exception(
            "run_type not recognized. warm or cold are only options"
        )

    if v > vlimit:
        raise Exception(f"Cannot set bias higher than {vlimit}")

    # 1. 先发送通用的全局初始化命令（只需执行一次）
    print("WR VBIASCTRL V 4095")
    response_data = device.command("WR VBIASCTRL V 4095")
    print(response_data)

    # 2. 循环遍历所有指定的 AFE 并设置电压
    for afe in afe_list:
        if v == 0:
            dac = 0
        else:
            dac = DAC_for_V(afe, v)

        CmdString = f"WR AFE {afe} BIASSET V {dac}"
        print(f"\n--- Setting AFE {afe} ---")
        print(CmdString)

        # 发送命令，读取响应
        response_data = device.command(CmdString)
        print(response_data)

    # 3. 最后读取所有监测数据
    print("\nRD VM ALL")
    print(device.command("RD VM ALL"))

    device.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPHNE command interpreter")
    parser.add_argument(
        "--ep", required=True, help="DAPHNE IP endpoint, e.g. 110"
    )

    # 这里的 type 接收逗号分隔字符串，并将其转换为 [0, 1, 2...] 的整型列表
    parser.add_argument(
        "--afe",
        required=True,
        type=lambda s: [int(item) for item in s.split(",")],
        help="Which AFE(s) are we setting bias for? e.g., 0 or 0,1,2,3,4",
    )

    parser.add_argument(
        "--v", required=True, help="What bias voltage do you want? Positive number"
    )
    parser.add_argument(
        "--run_type",
        required=True,
        help="Run type. Choose warm for warm test and cold for cold test.",
    )

    args = parser.parse_args()

    # 此时 args.afe 已经是一个包含整数的 list 了
    main(args.ep, args.afe, args.v, args.run_type)
