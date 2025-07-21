#!/bin/bash
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
# directory for log and checkpoints, including --checkpoint-dir, --online-policy-save-path, and --online-policy-path
mkdir -p log/$DATE

# EAGLE3 SB3 Discrete PPO Online RL: Training + Testing
# This script trains an SB3 discrete PPO policy, then tests it

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 SB3 Discrete PPO Online RL: Training + Testing Workflow ===" | tee log/$DATE/output_sb3_discrete_ppo.log
echo "Model: $MODEL_PATH" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Algorithm: Stable Baselines 3 Max-Entropy Discrete PPO with diverse parameter exploration" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Constraint: total_tokens ≤ top_k^(depth-1)" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "SB3 Features: Optimized PPO, vectorized environments, robust training, max-entropy exploration" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Max-Entropy Features: High entropy coefficient, temperature-based inference, diverse parameter sampling" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Workflow: Train → Test → Analyze" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Resume: Checkpoints → Fallback to final model → Fresh start" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Log Directory: log/$DATE" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log

# Training Phase: Train SB3 discrete PPO policy
echo "Phase 1: Training with SB3 Discrete PPO..." | tee -a log/$DATE/output_sb3_discrete_ppo.log
PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-sb3-discrete-ppo-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --online-lr 0.0003 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --ppo-clip-range 0.2 \
    --ppo-gamma 0.95 \
    --ppo-gae-lambda 0.9 \
    --ppo-ent-coef 0.1 \
    --ppo-vf-coef 0.5 \
    --inference-temperature 1.5 \
    --max-entropy-inference \
    --online-policy-save-path "log/$DATE/sb3_discrete_ppo_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE/sb3_discrete_ppo_checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 5 \
    --no-resume \
    2>&1 | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Phase 2: Testing with trained SB3 Discrete PPO policy (inference-only)..." | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Loading policy from: log/$DATE/sb3_discrete_ppo_policy.pth" | tee -a log/$DATE/output_sb3_discrete_ppo.log

# Check if the trained policy exists
if [ ! -f "log/$DATE/sb3_discrete_ppo_policy_sb3.zip" ]; then
    echo "❌ Error: log/$DATE/sb3_discrete_ppo_policy_sb3.zip not found!" | tee -a log/$DATE/output_sb3_discrete_ppo.log
    echo "   Make sure Phase 1 (training) completed successfully." | tee -a log/$DATE/output_sb3_discrete_ppo.log
    exit 1
fi

python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-sb3-discrete-ppo-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --online-policy-path "log/$DATE/sb3_discrete_ppo_policy.pth" \
    --online-inference-only \
    --inference-temperature 1.5 \
    --max-entropy-inference \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "log/$DATE/eagle-sb3-discrete-ppo-test-results.jsonl" \
    --no-wandb \
    2>&1 | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "=== Performance Analysis ===" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "1. SB3 Discrete PPO Policy vs EAGLE3 Baseline:" | tee -a log/$DATE/output_sb3_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-sb3-discrete-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "2. EAGLE3 Baseline vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_sb3_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "3. SB3 Discrete PPO Policy vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_sb3_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-sb3-discrete-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_sb3_discrete_ppo.log

echo "All results and logs saved to: log/$DATE/" | tee -a log/$DATE/output_sb3_discrete_ppo.log
echo "Checkpoints saved to: log/$DATE/sb3_discrete_ppo_checkpoints/" | tee -a log/$DATE/output_sb3_discrete_ppo.log
