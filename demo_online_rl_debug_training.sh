#!/bin/bash

# Demo script for EAGLE3 Online RL Training Mode
# This script trains the online RL policy with proper exploration and learning

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Online RL Training Demo ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Mode: Training with exploration and learning"
echo ""

# Run training with aggressive exploration to see parameter variation
# python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-online-rl-training" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-lr 0.01 \
#     --online-epsilon-start 0.9 \
#     --online-epsilon-end 0.3 \
#     --online-memory-size 50 \
#     --online-batch-size 4 \
#     --online-policy-save-path "online_training_policy.pth" \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 256 \
#     --num-choices 1 #\
    # --question-begin 0 \
    # --question-end 10

echo ""
echo "=== Training Complete ==="
echo "Policy saved to: online_training_policy.pth"
echo ""
echo "This training run should show parameter variation due to:"
echo "- High initial exploration (epsilon=0.9)"
echo "- Aggressive learning rate (0.01)"
echo "- Small memory for quick updates"
echo "- Real-time parameter optimization"

python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-online-rl-inference" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-lr 0.01 \
    --online-epsilon-start 0.9 \
    --online-epsilon-end 0.3 \
    --online-memory-size 50 \
    --online-batch-size 4 \
    --online-policy-save-path "online_training_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 256 \
    --num-choices 1 --online-inference-only

python eagle/evaluation/speed.py   --ea-file "mt_bench/eagle-online-rl-inference-temperature-0.0.jsonl"   --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl"   --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct"