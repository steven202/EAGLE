#!/bin/bash
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
# directory for log and checkpoints, including --checkpoint-dir, --online-policy-save-path, and --online-policy-path
mkdir -p log/$DATE

# EAGLE3 Online RL Expanded Action Space: Training + Testing
# This script trains a policy with expanded action space, then tests it

# Configuration
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE3 Online RL: Training + Testing Workflow ===" | tee log/$DATE/output_disc_space.log
echo "Model: $MODEL_PATH" | tee -a log/$DATE/output_disc_space.log
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/output_disc_space.log
echo "Action Space: 5×5×5 = 125 total (116 valid with constraints)" | tee -a log/$DATE/output_disc_space.log
echo "Constraint: total_tokens ≤ top_k^(depth-1)" | tee -a log/$DATE/output_disc_space.log
echo "Workflow: Train → Test → Analyze" | tee -a log/$DATE/output_disc_space.log
echo "Resume: Checkpoints → Fallback to final model → Fresh start" | tee -a log/$DATE/output_disc_space.log
echo "Log Directory: log/$DATE" | tee -a log/$DATE/output_disc_space.log
echo "" | tee -a log/$DATE/output_disc_space.log

# Training Phase: Train policy with expanded action space
echo "Phase 1: Training with expanded action space..." | tee -a log/$DATE/output_disc_space.log
PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
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
    --online-policy-save-path "log/$DATE/expanded_action_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE" \
    --online-repeat-factor 20 \
    --no-resume \
    2>&1 | tee -a log/$DATE/output_disc_space.log

echo "" | tee -a log/$DATE/output_disc_space.log
echo "Phase 2: Testing with trained policy (inference-only)..." | tee -a log/$DATE/output_disc_space.log
echo "Loading policy from: log/$DATE/expanded_action_policy.pth" | tee -a log/$DATE/output_disc_space.log

# Check if the trained policy exists
if [ ! -f "log/$DATE/expanded_action_policy.pth" ]; then
    echo "❌ Error: log/$DATE/expanded_action_policy.pth not found!" | tee -a log/$DATE/output_disc_space.log
    echo "   Make sure Phase 1 (training) completed successfully." | tee -a log/$DATE/output_disc_space.log
    exit 1
fi

python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-constraint-test" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --online-policy-path "log/$DATE/expanded_action_policy.pth" \
    --online-inference-only \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --answer-file "log/$DATE/eagle-constraint-test-results.jsonl" \
    2>&1 | tee -a log/$DATE/output_disc_space.log

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

echo "=== Performance Analysis ===" | tee -a log/$DATE/output_disc_space.log
echo "" | tee -a log/$DATE/output_disc_space.log

echo "1. Trained Policy vs EAGLE3 Baseline:" | tee -a log/$DATE/output_disc_space.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-constraint-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_disc_space.log

echo "" | tee -a log/$DATE/output_disc_space.log
echo "2. EAGLE3 Baseline vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_disc_space.log
python eagle/evaluation/speed.py \
    --ea-file "mt_bench/LLaMA3.1-8B_eagle3.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_disc_space.log

echo "" | tee -a log/$DATE/output_disc_space.log
echo "3. Trained Policy vs Standard LLaMA Baseline:" | tee -a log/$DATE/output_disc_space.log
python eagle/evaluation/speed.py \
    --ea-file "log/$DATE/eagle-constraint-test-results.jsonl" \
    --baseline-file "mt_bench/LLaMA3.1-8B_baseline.jsonl" \
    --tokenizer-path "meta-llama/Llama-3.1-8B-Instruct" \
    2>&1 | tee -a log/$DATE/output_disc_space.log

echo "" | tee -a log/$DATE/output_disc_space.log
echo "=== Analysis Complete ===" | tee -a log/$DATE/output_disc_space.log
echo "All results and logs saved to: log/$DATE/" | tee -a log/$DATE/output_disc_space.log