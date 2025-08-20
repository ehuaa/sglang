docker run --gpus=all \
    --shm-size 32g \
    --name sglang_oss_test \
    -p 30012:30012 \
    -e CUDA_VISIBLE_DEVICES="0,1,2,3" \
    -v /nas:/nas \
    -v /data:/root/models \
    --ipc=host \
    sglang:v0.4.10 \
    python3 -m sglang.launch_server --model-path /nas/xq/models/gpt-oss-120b --tp 2 --enable-mixed-chunk --mem-fraction-static 0.93 --host 0.0.0.0 --port 30012