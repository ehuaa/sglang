docker run --gpus=all \
    --shm-size 32g \
    --name sglang_fp8 \
    -p 30013:30013 \
    -e CUDA_VISIBLE_DEVICES="0,1,2,3" \
    -v /nas:/nas \
    -v /data:/root/models \
    --ipc=host \
    sglang:v0.5.1_fp8_base_vllm \
    python3 -m sglang.launch_server --model-path /nas/czh/Qwen3-30B-A3B-FP8/Qwen3-30B-A3B-FP8/ --tp 2 --enable-mixed-chunk --mem-fraction-static 0.93 --host 0.0.0.0 --port 30013