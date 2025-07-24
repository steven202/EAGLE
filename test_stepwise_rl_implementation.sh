#!/bin/bash

# Test script for Step-wise RL Implementation
# This tests the new step-wise RL functionality where parameters are predicted at each draft step

DATE=$(date '+%Y%m%d_%H%M')
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
LOG_DIR="log/stepwise_test_$DATE"

# Create log directory
mkdir -p $LOG_DIR

echo "=== Step-wise RL Implementation Test ===" | tee $LOG_DIR/test.log
echo "Date: $DATE" | tee -a $LOG_DIR/test.log
echo "Model: $MODEL_PATH" | tee -a $LOG_DIR/test.log  
echo "Base Model: $BASE_MODEL_PATH" | tee -a $LOG_DIR/test.log
echo "Test: Comparing traditional turn-wise vs new step-wise RL" | tee -a $LOG_DIR/test.log
echo "" | tee -a $LOG_DIR/test.log

# Test 1: Traditional Turn-wise RL (existing behavior)
echo "=== Test 1: Traditional Turn-wise RL (Control) ===" | tee -a $LOG_DIR/test.log
echo "Parameters predicted once per turn, policy updated once per turn" | tee -a $LOG_DIR/test.log

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-turnwise-rl-test-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --online-lr 0.001 \
    --ppo-n-steps 32 \
    --ppo-batch-size 16 \
    --ppo-epochs 2 \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 256 \
    --num-choices 1 \
    --online-policy-save-path "$LOG_DIR/turnwise_policy.pth" \
    --checkpoint-dir "$LOG_DIR/turnwise_checkpoints" \
    --checkpoint-freq 10 \
    --online-repeat-factor 1 \
    --question-begin 0 --question-end 5 \
    --no-wandb \
    2>&1 | tee -a $LOG_DIR/turnwise_training.log

echo "" | tee -a $LOG_DIR/test.log

# Test 2: New Step-wise RL (new implementation)
echo "=== Test 2: Step-wise RL (New Implementation) ===" | tee -a $LOG_DIR/test.log
echo "Parameters predicted at each draft step, policy updated at each step" | tee -a $LOG_DIR/test.log

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-stepwise-rl-test-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --use-stepwise-rl \
    --stepwise-reward-type tokens_per_second \
    --online-lr 0.001 \
    --ppo-n-steps 32 \
    --ppo-batch-size 16 \
    --ppo-epochs 2 \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 256 \
    --num-choices 1 \
    --online-policy-save-path "$LOG_DIR/stepwise_policy.pth" \
    --checkpoint-dir "$LOG_DIR/stepwise_checkpoints" \
    --checkpoint-freq 10 \
    --online-repeat-factor 1 \
    --question-begin 0 --question-end 5 \
    --no-wandb \
    2>&1 | tee -a $LOG_DIR/stepwise_training.log

echo "" | tee -a $LOG_DIR/test.log

# Test 3: Error validation - Step-wise RL without online RL (should fail)
echo "=== Test 3: Error Validation - Step-wise RL without Online RL ===" | tee -a $LOG_DIR/test.log
echo "This should fail with an error message" | tee -a $LOG_DIR/test.log

python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-stepwise-error-test-$DATE" \
    --bench-name "mt_bench" \
    --use-stepwise-rl \
    --question-begin 0 --question-end 1 \
    2>&1 | tee -a $LOG_DIR/error_test.log

echo "" | tee -a $LOG_DIR/test.log

# Test 4: Step-wise RL inference mode (no training)
echo "=== Test 4: Step-wise RL Inference Mode ===" | tee -a $LOG_DIR/test.log
echo "Using trained step-wise policy for inference only" | tee -a $LOG_DIR/test.log

if [ -f "$LOG_DIR/stepwise_policy_sb3.zip" ]; then
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --base-model-path "$BASE_MODEL_PATH" \
        --ea-model-path "$MODEL_PATH" \
        --model-id "eagle-stepwise-inference-test-$DATE" \
        --bench-name "mt_bench" \
        --use-online-rl \
        --use-sb3-discrete-ppo \
        --use-stepwise-rl \
        --online-policy-path "$LOG_DIR/stepwise_policy.pth" \
        --online-inference-only \
        --temperature 0.0 \
        --use_eagle3 \
        --max-new-token 256 \
        --num-choices 1 \
        --question-begin 0 --question-end 3 \
        --no-wandb \
        2>&1 | tee -a $LOG_DIR/stepwise_inference.log
else
    echo "⚠️ Stepwise policy not found, skipping inference test" | tee -a $LOG_DIR/test.log
fi

echo "" | tee -a $LOG_DIR/test.log
echo "=== Test Summary ===" | tee -a $LOG_DIR/test.log
echo "1. Turn-wise RL: $([ -f "$LOG_DIR/turnwise_training.log" ] && echo "✅ Completed" || echo "❌ Failed")" | tee -a $LOG_DIR/test.log
echo "2. Step-wise RL: $([ -f "$LOG_DIR/stepwise_training.log" ] && echo "✅ Completed" || echo "❌ Failed")" | tee -a $LOG_DIR/test.log
echo "3. Error validation: $([ -f "$LOG_DIR/error_test.log" ] && echo "✅ Completed" || echo "❌ Failed")" | tee -a $LOG_DIR/test.log
echo "4. Step-wise inference: $([ -f "$LOG_DIR/stepwise_inference.log" ] && echo "✅ Completed" || echo "❌ Failed")" | tee -a $LOG_DIR/test.log

echo "" | tee -a $LOG_DIR/test.log
echo "Logs saved to: $LOG_DIR" | tee -a $LOG_DIR/test.log
echo "Key differences to look for:" | tee -a $LOG_DIR/test.log
echo "- Turn-wise: 'RL params' printed once per question" | tee -a $LOG_DIR/test.log
echo "- Step-wise: 'Step X RL params' printed multiple times per question" | tee -a $LOG_DIR/test.log
echo "- Step-wise: 'Step X reward' messages during training" | tee -a $LOG_DIR/test.log

# Make script executable
chmod +x $0
