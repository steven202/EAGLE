#!/bin/bash

# EAGLE3 Continuous Action Space Online RL: Training + Testing + Performance Analysis
# This script trains a continuous policy, tests it, and compares performance with discrete baseline

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Continuous Online RL: Performance Testing Workflow ==="
echo "Model: $MODEL_PATH"
echo "Base Model: $BASE_MODEL_PATH"
echo "Action Space: Continuous (total_tokens: 16-128, depth: 2-8, top_k: 2-32)"
echo "Constraint: total_tokens ≤ top_k^(depth-1) (enforced automatically)"
echo "Workflow: Train Continuous → Test Continuous → Compare with Discrete → Analyze Performance"
echo ""

# Phase 1: Training with continuous action space
echo "Phase 1: Training with continuous action space..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-continuous-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --continuous-action-space \
    --online-lr 0.001 \
    --online-epsilon-start 0.3 \
    --online-epsilon-end 0.02 \
    --online-memory-size 100 \
    --online-batch-size 8 \
    --online-policy-save-path "continuous_action_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-freq 20 \
    --checkpoint-dir "checkpoints_continuous" \
    --no-resume 

echo ""
echo "Phase 2: Testing with trained continuous policy (inference-only)..."
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-continuous-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --continuous-action-space \
    --online-policy-path "continuous_action_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "mt_bench/eagle-continuous-test-results.jsonl"

# echo ""
# echo "Phase 3: Training discrete baseline for comparison..."
# python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-discrete-training" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-lr 0.01 \
#     --online-epsilon-start 0.9 \
#     --online-epsilon-end 0.3 \
#     --online-memory-size 100 \
#     --online-batch-size 8 \
#     --online-policy-save-path "discrete_action_policy.pth" \
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
#     --model-id "eagle-discrete-test" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-policy-path "discrete_action_policy.pth" \
#     --online-inference-only \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --answer-file "mt_bench/eagle-discrete-test-results.jsonl"

echo ""
echo "=== Performance Analysis ==="

# echo ""
# echo "Performance Comparison: Continuous vs Discrete vs Baseline..."

# Compare continuous action space with discrete baseline
# echo "1. Continuous Action Space vs Discrete Action Space:"
# python eagle/evaluation/speed.py \
#     --ea-file "mt_bench/eagle-continuous-test-results.jsonl" \
#     --baseline-file "mt_bench/eagle-discrete-test-results.jsonl" \
#     --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "2. Continuous Action Space vs EAGLE3 Baseline:"
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/eagle-continuous-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH"

# echo ""
# echo "3. Discrete Action Space vs EAGLE3 Baseline:"
# python eagle/evaluation/speed.py \
#     --ea-file "mt_bench/eagle-discrete-test-results.jsonl" \
#     --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
#     --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "4. Continuous Action Space vs Standard LLaMA Baseline:"
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/eagle-continuous-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH"

echo ""
echo "5. EAGLE3 Baseline vs Standard LLaMA Baseline:"
python eagle/evaluation/speed.py   --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl"   --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl"   --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct"

