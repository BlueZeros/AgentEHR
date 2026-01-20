MODEL="/sfs/data/ShareModels/LLMs/Qwen3-30B-A3B-Instruct-2507"
CUDA_ID=${1:-0}

echo "run vllm on cuda: ${CUDA_ID}, port: $((4000 + CUDA_ID))"

CUDA_VISIBLE_DEVICES=${CUDA_ID} vllm serve ${MODEL} \
    --port $((4000 + CUDA_ID)) \
    --dtype auto  \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 64000 
    # --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${MODEL} \
#     --port $((4000 + CUDA_ID)) \
#     --dtype auto  \
#     --tensor-parallel-size 4 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --reasoning-parser qwen3 \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 64000