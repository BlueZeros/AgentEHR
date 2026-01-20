#!/bin/bash

# EHR Agent Benchmark Testing Script

SPLIT="train"
TASK="mix_600"

DATA_ROOT="/home/ma-user/work/liaoyusheng/projects/EHRAgent/datas/EHRAgentBench/${SPLIT}"
DATA_NAME="${DATA_ROOT}/${TASK}.json"
OUTPUT_PATH="../../ckpt/${SPLIT}_${TASK}/Qwen3-30B-A3B-Instruct-2507"

python optimization_mcp.py \
    --data_path "$DATA_NAME" \
    --output_path "$OUTPUT_PATH" \
    --model_name_or_path qwen3_30b_moe \
    --vllm_server_url "http://127.0.0.1:8000" \
    --mcp_url "http://127.0.0.1:9000/mcp" \
    --temperature 0.7 \
    --agent_type "mcp_reflectool" \
    --search_size 1 \
    --batch 4 \
    --max_step 100 \
    --resume