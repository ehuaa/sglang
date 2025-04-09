#!/bin/bash

if [ -z "$RANK" ]; then
    echo "ERROR: RANK environment variable is not set."
    exit 1
fi

# 直接根据 RANK 执行不同命令
if [ "$RANK" -eq 0 ]; then
    python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --tp 32 \
        --dist-init-addr ${MASTER_ADDR}:5000 \
        --nnodes 4 \
        --node-rank "$RANK" \
        --trust-remote-code \
        --enable-flashinfer-mla \
        --chunked-prefill-size 16384 \
        --enable-mixed-chunk \
        --host 0.0.0.0 \
        --port 30000
else
     python3 -m sglang.launch_server \
        --model-path "$MODEL_DIR" \
        --tp 32 \
        --dist-init-addr ${MASTER_ADDR}:5000 \
        --nnodes 4 \
        --node-rank "$RANK" \
        --trust-remote-code \
        --enable-flashinfer-mla \
        --chunked-prefill-size 16384 \
        --enable-mixed-chunk
fi