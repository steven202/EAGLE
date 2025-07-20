#!/bin/bash

# EAGLE3 Parameter Combination Testing
# Tests different parameter combinations on same questions to find consistently optimal parameters

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Parameter Combination Testing ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Objective: Find consistently best (fastest) parameter combinations"
echo ""

echo "Testing parameter combinations..."
PYTHONUNBUFFERED=1 python -m eagle.evaluation."gen_ea_answer_llama3chat_rl_test_opt" \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-param-test" \
    --bench-name "mt_bench" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --online-repeat-factor 1\
    --num-choices 1 #\
    # --question-begin 0 \
    # --question-end 5

# echo ""
# echo "=== Parameter Testing Complete ==="
# echo "This test helps identify:"
# echo "✓ Which parameter combinations are consistently fastest"
# echo "✓ Whether optimal parameters are question-dependent"
# echo "✓ Performance variation across different combinations"
# echo "✓ Error rates for different parameter ranges"
