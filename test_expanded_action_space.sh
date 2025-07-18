#!/bin/bash

# EAGLE3 Online RL Expanded Action Space: Training + Testing
# This script trains a policy with expanded action space, then tests it

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Online RL: Training + Testing Workflow ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Action Space: 5×5×5 = 125 total (116 valid with constraints)"
echo "Constraint: total_tokens ≤ top_k^(depth-1)"
echo "Workflow: Train → Test → Analyze"
echo ""

# Training Phase: Train policy with expanded action space
echo "Phase 1: Training with expanded action space..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-constraint-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-lr 0.01 \
    --online-epsilon-start 0.9 \
    --online-epsilon-end 0.3 \
    --online-memory-size 100 \
    --online-batch-size 8 \
    --online-policy-save-path "expanded_action_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 #\
    # --question-begin 0 \
    # --question-end 40

echo ""
echo "Phase 2: Testing with trained policy (inference-only)..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-constraint-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-policy-path "expanded_action_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "mt_bench/eagle-constraint-test-results.jsonl"

# echo ""
# echo "=== Training and Testing Complete ==="
# echo "Training results: mt_bench/eagle-constraint-training-temperature-0.0.jsonl"
# echo "Testing results: mt_bench/eagle-constraint-test-results.jsonl"
# echo "Trained policy: expanded_action_policy.pth"
# echo ""
# echo "Verification checklist:"
# echo "✓ Training phase should show varied parameters with ✓ marks"
# echo "✓ Testing phase should use learned policy without training"
# echo "✓ All combinations should satisfy: tt ≤ k^(d-1)"
# echo "✓ Parameter diversity should be visible in both phases"
# echo ""
# echo "Parameter ranges validated:"
# echo "- total_tokens: [32, 48, 64, 80, 96]"
# echo "- depth: [3, 4, 5, 6, 7]" 
# echo "- top_k: [4, 8, 12, 16, 20]"
# echo "- Constraint safety: 116/125 valid combinations (92.8%)"

python eagle/evaluation/speed.py   --ea-file "mt_bench/eagle-constraint-test-results.jsonl"   --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl"   --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct"
# Speed ratio (EAGLE/Baseline): 0.9777922787100434 times faster