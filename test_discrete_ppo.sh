#!/bin/bash
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
# directory for log and checkpoints, including --checkpoint-dir, --online-policy-save-path, and --online-policy-path
mkdir -p log/$DATE

# EAGLE3 Discrete PPO Online RL: Training + Testing
# This script trains a discrete PPO policy, then tests it

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Discrete PPO Online RL: Training + Testing Workflow ===" | tee log/$DATE/output_discrete_ppo.log
echo "Model: $MODEL_PATH" | tee -a log/$DATE/output_discrete_ppo.log
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/output_discrete_ppo.log
echo "Algorithm: Discrete PPO with Actor-Critic" | tee -a log/$DATE/output_discrete_ppo.log
echo "Constraint: total_tokens ≤ top_k^(depth-1)" | tee -a log/$DATE/output_discrete_ppo.log
echo "PPO Features: GAE, clipped surrogate loss, value function learning" | tee -a log/$DATE/output_discrete_ppo.log
echo "Workflow: Train → Test → Analyze" | tee -a log/$DATE/output_discrete_ppo.log
echo "Resume: Checkpoints → Fallback to final model → Fresh start" | tee -a log/$DATE/output_discrete_ppo.log
echo "Log Directory: log/$DATE" | tee -a log/$DATE/output_discrete_ppo.log
echo "" | tee -a log/$DATE/output_discrete_ppo.log

# Training Phase: Train discrete PPO policy
echo "Phase 1: Training with Discrete PPO..." | tee -a log/$DATE/output_discrete_ppo.log
PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-discrete-ppo-training" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-discrete-ppo \
    --online-lr 0.0001 \
    --ppo-epochs 4 \
    --ppo-batch-size 32 \
    --ppo-clip-range 0.2 \
    --ppo-gamma 0.99 \
    --ppo-gae-lambda 0.95 \
    --online-policy-save-path "log/$DATE/discrete_ppo_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE/discrete_ppo_checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 5 \
    --no-resume \
    2>&1 | tee -a log/$DATE/output_discrete_ppo.log

echo "" | tee -a log/$DATE/output_discrete_ppo.log
echo "Phase 2: Testing with trained Discrete PPO policy (inference-only)..." | tee -a log/$DATE/output_discrete_ppo.log
echo "Loading policy from: log/$DATE/discrete_ppo_policy.pth" | tee -a log/$DATE/output_discrete_ppo.log

# Check if the trained policy exists
if [ ! -f "log/$DATE/discrete_ppo_policy.pth" ]; then
    echo "❌ Error: log/$DATE/discrete_ppo_policy.pth not found!" | tee -a log/$DATE/output_discrete_ppo.log
    echo "   Make sure Phase 1 (training) completed successfully." | tee -a log/$DATE/output_discrete_ppo.log
    exit 1
fi

python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-discrete-ppo-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-discrete-ppo \
    --online-policy-path "log/$DATE/discrete_ppo_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "log/$DATE/eagle-discrete-ppo-test-results.jsonl" \
    --no-wandb \
    2>&1 | tee -a log/$DATE/output_discrete_ppo.log

echo "" | tee -a log/$DATE/output_discrete_ppo.log
echo "=== Performance Analysis ===" | tee -a log/$DATE/output_discrete_ppo.log
echo "" | tee -a log/$DATE/output_discrete_ppo.log

echo "1. Discrete PPO Policy vs EAGLE3 Baseline:" | tee -a log/$DATE/output_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-discrete-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_discrete_ppo.log

echo "" | tee -a log/$DATE/output_discrete_ppo.log
echo "2. EAGLE3 Baseline vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_discrete_ppo.log

echo "" | tee -a log/$DATE/output_discrete_ppo.log
echo "3. Discrete PPO Policy vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_discrete_ppo.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-discrete-ppo-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_discrete_ppo.log

echo "All results and logs saved to: log/$DATE/" | tee -a log/$DATE/output_discrete_ppo.log
echo "Checkpoints saved to: log/$DATE/discrete_ppo_checkpoints/" | tee -a log/$DATE/output_discrete_ppo.log
