docker run --gpus=all \
    --shm-size 32g \
    --name sglang_test \
    -p 30012:30012 \
    -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    -v /nas:/nas \
    -v /data:/root/models \
    --ipc=host \
    sglang:v0.4.6-post5 \
    python3 -m sglang.launch_server --model-path /nas/xq/models/geogpt-r1/geogpt-q25-72b-cpt-r1-s2-dpo-0423 --tp 8 --trust-remote-code --host 0.0.0.0 --port 30012 --attention-backend fa3 --enable-mixed-chunk --reasoning-parser deepseek-r1