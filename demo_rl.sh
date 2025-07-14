#!/bin/bash

# Real EAGLE RL parameter optimization evaluation script
# Trains RL policy and compares performance with fixed parameters on full MT-Bench dataset

# Create log file with timestamp
DATE=$(date '+%Y%m%d_%H%M')
echo "Run date: $DATE"
mkdir -p log/$DATE
LOG_FILE="log/$DATE/demo_rl.log"

trap 'echo "Caught Ctrl+C, stopping..."; pkill -P $$; exit 1' SIGINT

# Function to log messages to both console and file
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to run command with logging
run_with_log() {
    echo "Running: $1" | tee -a "$LOG_FILE"
    eval "$1" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log_message "❌ Command failed with exit code $exit_code"
        return $exit_code
    fi
    return 0
}

log_message "=== EAGLE RL Parameter Optimization Evaluation ==="
log_message "This script runs a complete evaluation of RL-based parameter optimization"
log_message "Training and evaluating on the full MT-Bench dataset"
log_message "Log file: $LOG_FILE"
log_message ""

# Check if required dependencies are installed
log_message "Step 1: Checking RL dependencies..."
python -c "import stable_baselines3, sentence_transformers, gymnasium, shimmy" 2>/dev/null
if [ $? -ne 0 ]; then
    log_message "Missing RL dependencies. Please install with:"
    log_message "pip install -r requirements-rl.txt"
    log_message "Or install individually:"
    log_message "pip install 'stable-baselines3>=2.0.0' 'gymnasium>=0.26.0' 'shimmy>=2.0.0' 'sentence-transformers>=2.2.0'"
    exit 1
fi
log_message "✓ All dependencies available"
log_message ""

# Set paths (modify these according to your setup)
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"  # Example: Update this path
EA_MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"   # Example: Update this path
BENCH_NAME="mt_bench"

log_message "Step 2: Training initial RL policy from questions..."
run_with_log "python -m eagle.evaluation.train_rl_policy \
    --mode from_questions \
    --question-file eagle/data/${BENCH_NAME}/question.jsonl \
    --policy-path ppo_tree_policy_initial.zip \
    --total-timesteps 2000"

if [ $? -ne 0 ]; then
    log_message "❌ Step 2 failed. Exiting."
    exit 1
fi

log_message "Step 3: Collecting real performance data..."
run_with_log "python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path ${BASE_MODEL_PATH} \
    --ea-model-path ${EA_MODEL_PATH} \
    --model-id eagle-rl-data-collection \
    --bench-name ${BENCH_NAME} \
    --collect-rl-data \
    --rl-data-file rl_training_data.jsonl \
    --temperature 0.0 \
    --use_eagle3"

if [ $? -ne 0 ]; then
    log_message "❌ Step 3 failed. Exiting."
    exit 1
fi

log_message "Step 4: Training refined RL policy from collected data..."
run_with_log "python -m eagle.evaluation.train_rl_policy \
    --mode from_data \
    --data-file rl_training_data.jsonl \
    --policy-path ppo_tree_policy_refined.zip \
    --total-timesteps 3000"

if [ $? -ne 0 ]; then
    log_message "❌ Step 4 failed. Exiting."
    exit 1
fi

log_message "Step 5: Running evaluation with trained RL policy..."
run_with_log "python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path ${BASE_MODEL_PATH} \
    --ea-model-path ${EA_MODEL_PATH} \
    --model-id eagle-rl-optimized \
    --bench-name ${BENCH_NAME} \
    --use-rl-policy \
    --rl-policy-path ppo_tree_policy_refined.zip \
    --temperature 0.0 \
    --use_eagle3"

if [ $? -ne 0 ]; then
    log_message "❌ Step 5 failed. Exiting."
    exit 1
fi

log_message "Step 6: Comparing results..."
log_message "Check the generated answer files:"
log_message "- ${BENCH_NAME}/eagle-rl-data-collection-temperature-0.0.jsonl (fixed default parameters)"
log_message "- ${BENCH_NAME}/eagle-rl-optimized-temperature-0.0.jsonl (RL-optimized parameters)"
log_message ""
log_message "Check the RL training data:"
log_message "- rl_training_data.jsonl (collected performance metrics)"
log_message ""
log_message "Real EAGLE RL evaluation completed! 🚀"
log_message "The RL policy has learned to optimize tree parameters dynamically."
log_message "Full log saved to: $LOG_FILE"