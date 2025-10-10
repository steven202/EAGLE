#!/bin/bash

# Test script for EAGLE RL Profiling
# This script runs a minimal test to verify the profiling functionality

echo "🚀 Testing EAGLE RL Profiling Script"
echo "====================================="

# Set up environment
export PYTHONPATH=/home/guo/EAGLE_RL_latency:$PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create test output directory
DATE=$(date +%Y%m%d_%H%M%S)
TEST_DIR="log/test_profiling_$DATE"
mkdir -p "$TEST_DIR/checkpoints"

echo "📁 Test output directory: $TEST_DIR"

# Create initial log file and add header
echo "🚀 Testing EAGLE RL Profiling Script" > "$TEST_DIR/profiling_output.log"
echo "=====================================" >> "$TEST_DIR/profiling_output.log"
echo "Test started at: $(date)" >> "$TEST_DIR/profiling_output.log"
echo "📁 Test output directory: $TEST_DIR" >> "$TEST_DIR/profiling_output.log"
echo "" >> "$TEST_DIR/profiling_output.log"

# Run profiling with minimal settings for quick test
echo "🔄 Running profiling test (80 questions only)..." | tee -a "$TEST_DIR/profiling_output.log"

conda run -n eagle-rl python -m eagle.evaluation.gen_ea_answer_llama3chat_rl_profiling \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --model-id optimized_max_entropy_ppo_profiling_test \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end 80 \
    --answer-file "$TEST_DIR/test_answers.jsonl" \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-sb3-discrete-ppo \
    --optimized-policy-version ofl \
    --online-lr 3e-4 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --ppo-gamma 0.95 \
    --ppo-gae-lambda 0.9 \
    --ppo-clip-range 0.2 \
    --ppo-vf-coef 0.5 \
    --ppo-ent-coef 0.01 \
    --max-grad-norm 0.5 \
    --enable-max-entropy \
    --max-entropy-ent-coef 0.1 \
    --inference-temperature 1.5 \
    --max-entropy-inference \
    --action-cache-steps 10 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --ppo-net-arch "128,128" \
    --checkpoint-dir "$TEST_DIR/checkpoints" \
    --online-policy-save-path "$TEST_DIR/test_policy.zip" \
    --checkpoint-freq 1 \
    --wandb-project eagle-profiling-test \
    --no-wandb \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-stepwise-rl \
    --use-eagle3 2>&1 | tee -a "$TEST_DIR/profiling_output.log"

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Profiling test completed successfully!" | tee -a "$TEST_DIR/profiling_output.log"
    echo "📊 Check the detailed breakdown report above." | tee -a "$TEST_DIR/profiling_output.log"
    echo "📁 Test outputs saved to: $TEST_DIR" | tee -a "$TEST_DIR/profiling_output.log"
    
    # Check if answer file was created
    if [ -f "$TEST_DIR/test_answers.jsonl" ]; then
        echo "✅ Answer file created successfully" | tee -a "$TEST_DIR/profiling_output.log"
        echo "📝 Number of answers: $(wc -l < "$TEST_DIR/test_answers.jsonl")" | tee -a "$TEST_DIR/profiling_output.log"
    else
        echo "⚠️  Answer file not found" | tee -a "$TEST_DIR/profiling_output.log"
    fi
    
    # Check if policy was saved
    if [ -f "$TEST_DIR/test_policy.zip" ]; then
        echo "✅ Policy file saved successfully" | tee -a "$TEST_DIR/profiling_output.log"
    else
        echo "⚠️  Policy file not found" | tee -a "$TEST_DIR/profiling_output.log"
    fi
    
else
    echo "❌ Profiling test failed!" | tee -a "$TEST_DIR/profiling_output.log"
    echo "🔍 Check the error messages above for debugging" | tee -a "$TEST_DIR/profiling_output.log"
    exit 1
fi

echo "" | tee -a "$TEST_DIR/profiling_output.log"
echo "🎯 Profiling test completed!" | tee -a "$TEST_DIR/profiling_output.log"
echo "   Use the detailed breakdown report to identify performance bottlenecks." | tee -a "$TEST_DIR/profiling_output.log"
