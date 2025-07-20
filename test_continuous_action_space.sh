#!/bin/bash

# EAGLE3 Continuous Action Space Online RL: Training + Testing + Performance Analysis
# This script trains a continuous policy, tests it, and compares performance with discrete baseline

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# Setup date-based logging
DATE=`date '+%Y%m%d_%H%M'`
mkdir -p log/$DATE

echo "=== EAGLE3 Continuous Online RL: Performance Testing Workflow ===" | tee log/$DATE/output_continuous_space.log
echo "Model: $MODEL_PATH" | tee -a log/$DATE/output_continuous_space.log
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/output_continuous_space.log
echo "Action Space: Continuous (total_tokens: 16-128, depth: 2-8, top_k: 2-32)" | tee -a log/$DATE/output_continuous_space.log
echo "Constraint: total_tokens ≤ top_k^(depth-1) (enforced automatically)" | tee -a log/$DATE/output_continuous_space.log
echo "Workflow: Train Continuous → Test Continuous → Compare with Discrete → Analyze Performance" | tee -a log/$DATE/output_continuous_space.log
echo "" | tee -a log/$DATE/output_continuous_space.log

# Phase 1: Training with continuous action space
echo "Phase 1: Training with continuous action space..." | tee -a log/$DATE/output_continuous_space.log
PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
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
    --online-policy-save-path "log/$DATE/continuous_action_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-freq 20 \
    --online-repeat-factor 20 \
    --checkpoint-dir "log/$DATE" \
    --no-resume \
    2>&1 | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "Phase 2: Testing with trained continuous policy (inference-only)..." | tee -a log/$DATE/output_continuous_space.log
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-continuous-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --continuous-action-space \
    --online-policy-path "log/$DATE/continuous_action_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "log/$DATE/eagle-continuous-test-results.jsonl" \
    2>&1 | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "=== Performance Analysis ===" | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "2. Continuous Action Space vs EAGLE3 Baseline:" | tee -a log/$DATE/output_continuous_space.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-continuous-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH" \
    2>&1 | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "4. Continuous Action Space vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_continuous_space.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-continuous-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "$BASE_MODEL_PATH" \
    2>&1 | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "5. EAGLE3 Baseline vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_continuous_space.log
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_continuous_space.log

echo "" | tee -a log/$DATE/output_continuous_space.log
echo "=== Analysis Complete ===" | tee -a log/$DATE/output_continuous_space.log
echo "All results and logs saved to: log/$DATE/" | tee -a log/$DATE/output_continuous_space.log

