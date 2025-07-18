#!/bin/bash

# Demo script for EAGLE3 Online RL Training Mode
# This script trains the online RL policy with proper exploration and learning

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Online RL Training Demo ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Mode: Training with expanded action space (125 → 116 valid combinations)"
echo "Action Space: total_tokens=[32,48,64,80,96], depth=[3,4,5,6,7], top_k=[4,8,12,16,20]"
echo "Safety Constraint: total_tokens ≤ top_k^(depth-1)"
echo ""

# Run training with aggressive exploration to see parameter variation
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-online-rl-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-lr 0.01 \
    --online-epsilon-start 0.9 \
    --online-epsilon-end 0.3 \
    --online-memory-size 100 \
    --online-batch-size 8 \
    --online-policy-save-path "online_training_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1

echo ""
echo "=== Training Complete ==="
echo "Policy saved to: online_training_policy.pth"
echo ""
echo "This training run demonstrates:"
echo "- Expanded action space: 125 total combinations (116 valid)"
echo "- Safety constraints: total_tokens ≤ top_k^(depth-1)"  
echo "- High exploration: epsilon 0.9 → 0.3 for parameter variation"
echo "- Aggressive learning: 0.01 learning rate for quick adaptation"
echo "- Runtime safety: Only valid parameter combinations are used"

# Run inference with the trained policy to see how it performs
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-online-rl-inference" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-policy-path "online_training_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1

python eagle/evaluation/speed.py \
    --ea-file "mt_bench/eagle-online-rl-inference-temperature-0.0.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct"