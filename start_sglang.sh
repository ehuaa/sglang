docker run --gpus all \
    --shm-size 32g \
    --name sglang_test \
    -p 30012:30012 \
    -v /nas/czh:/root/czh \
    -v /data:/root/models \
    --gpus=all \
    --ipc=host \
    sglang/v0.4.2-post4:latest \
    python3 -m sglang.launch_server --model-path /root/models/xq/qwen2-5-72b-dpo-1101 --tp 8 --trust-remote-code --host 0.0.0.0 --port 30012