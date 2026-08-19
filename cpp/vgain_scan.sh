#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

RUN=$1

CH=0
EP=110
AFE=0
BASELINE=8000

for vgain in $(seq 0 100 3000)
do
    echo "=================================="
    echo "Run ${RUN} | CH=${CH} | vgain=${vgain}"
    echo "=================================="

    # ===== 设置硬件 =====
    python3 ../setup_daphne/offset_tuning.py \
        --ep ${EP} \
        --ch ${CH} \
        --afe ${AFE} \
        --baseline ${BASELINE} \
        --vgain ${vgain} || exit 1

    # ===== 采数据（run 只能是数字）=====
    ./collect_data \
        --trig ext \
        --ep ${EP} \
        --nwvfms 2000 \
        --afe ${AFE} \
        --ch ${CH} \
        --run ${RUN} || exit 1

    # ===== 重命名（加入 vgain）=====
    oldfile="data/run${RUN}_ch${CH}.hdf5"
    newfile="data/run${RUN}_ch${CH}_vgain${vgain}.hdf5"

    if [ -f "$oldfile" ]; then
        mv "$oldfile" "$newfile"
    else
        echo "⚠️ File $oldfile not found!"
        exit 1
    fi

    sleep 1
done

echo "✅ vgain scan finished."
