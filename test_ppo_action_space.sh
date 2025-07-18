#!/bin/bash

# EAGLE3 PPO Online RL: Training + Testing + Performance Analysis
# This script tests the PPO-based continuous action space policy

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 PPO Online RL: Performance Testing Workflow ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Algorithm: PPO (Proximal Policy Optimization)"
echo "Action Space: Continuous (total_tokens: 16-128, depth: 2-8, top_k: 2-32)"
echo "Constraint: total_tokens ≤ top_k^(depth-1) (enforced automatically)"
echo "Advantages: More stable learning, better sample efficiency, proven performance"
echo ""

# Check if stable-baselines3 is installed
python -c "import stable_baselines3; print('✅ Stable Baselines3 available')" 2>/dev/null || {
    echo "❌ Stable Baselines3 not found. Installing..."
    pip install stable-baselines3[extra]
    echo "✅ Stable Baselines3 installed"
}

# Phase 1: Training with PPO
echo "Phase 1: Training with PPO algorithm..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-ppo-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-ppo \
    --online-lr 0.0003 \
    --ppo-n-steps 1024 \
    --ppo-batch-size 32 \
    --ppo-n-epochs 4 \
    --ppo-gamma 0.99 \
    --ppo-gae-lambda 0.95 \
    --ppo-clip-range 0.2 \
    --online-policy-save-path "ppo_action_policy.zip" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-freq 20 \
    --checkpoint-dir "checkpoints_ppo" \
    --no-resume

echo ""
echo "Phase 2: Testing with trained PPO policy (inference-only)..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-ppo-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-ppo \
    --online-policy-path "ppo_action_policy.zip" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "mt_bench/eagle-ppo-test-results.jsonl"

# echo ""
# echo "Phase 3: Training discrete baseline for comparison..."
# python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-discrete-baseline" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-lr 0.01 \
#     --online-epsilon-start 0.9 \
#     --online-epsilon-end 0.3 \
#     --online-memory-size 100 \
#     --online-batch-size 8 \
#     --online-policy-save-path "discrete_baseline_policy.pth" \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --checkpoint-freq 20 \
#     --no-resume

# echo ""
# echo "Phase 4: Testing discrete baseline (inference-only)..."
# python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-discrete-baseline-test" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-policy-path "discrete_baseline_policy.pth" \
#     --online-inference-only \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --answer-file "mt_bench/eagle-discrete-baseline-test-results.jsonl"

echo ""
echo "=== Performance Analysis ==="

echo ""
echo "Performance Comparison: PPO vs Discrete vs Baseline..."

# Compare PPO with discrete baseline
# echo "1. PPO vs Discrete Action Space:"
# python eagle/evaluation/speed.py \
#     --ea-file "mt_bench/eagle-ppo-test-results.jsonl" \
#     --baseline-file "mt_bench/eagle-discrete-baseline-test-results.jsonl" \
#     --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "2. PPO vs EAGLE3 Baseline:"
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/eagle-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH"

# echo ""
# echo "3. Discrete vs EAGLE3 Baseline:"
# python eagle/evaluation/speed.py \
#     --ea-file "mt_bench/eagle-discrete-baseline-test-results.jsonl" \
#     --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
#     --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "4. PPO vs Standard LLaMA Baseline:"
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/eagle-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "6. LLAMA3.1 EAGLE3 vs Standard LLaMA Baseline:"
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct"
