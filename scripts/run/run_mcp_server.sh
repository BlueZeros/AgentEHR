
GPU_ID=${1:-0}

DATA_PATH="../data/EHRAgentBench"
HOST=127.0.0.1
PORT=5000

CUDA_VISIBLE_DEVICES=${GPU_ID} python run_mcp_server.py \
        --mode "http" \
        --host $HOST \
        --port $PORT \
        --data_path "$DATA_PATH"