#!/bin/bash

# Test script for Optimized RL Policy with Layer Feature Concatenation and Reduced Action Frequency
# This demonstrates the two key optimizations for faster RL inference

DATE=$(date '+%Y%m%d_%H%M')
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=50  # Small test set

# Create log directory
mkdir -p log/${DATE}_optimized_rl_test/{standard,optimized,comparison}

echo "=== EAGLE Optimized RL Policy Test ===" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Comparing Standard vs Optimized RL Policy Performance" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Model: $MODEL_PATH" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Test Questions: 0-$QUESTION_END" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log

echo "=== Phase 1: Testing Standard Online RL Policy (Baseline) ===" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Uses SBERT embeddings for state (384 dimensions)" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Generates action at every step" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Record start time
start_time=$(date +%s)

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-standard-rl-test-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-stepwise-rl \
    --online-lr 0.001 \
    --online-policy-save-path "log/${DATE}_optimized_rl_test/standard/standard_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 256 \
    --num-choices 1 \
    --checkpoint-dir "log/${DATE}_optimized_rl_test/standard/checkpoints" \
    --checkpoint-freq 25 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    --answer-file "log/${DATE}_optimized_rl_test/standard/answers.jsonl" \
    2>&1 | tee -a log/${DATE}_optimized_rl_test/standard/output.log

# Record end time
end_time=$(date +%s)
standard_duration=$((end_time - start_time))

echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "=== Phase 2: Testing Optimized RL Policy ===" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Optimization 1: Layer Feature Concatenation" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Uses EAGLE-3's 3k-dimensional concatenated layer features" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- No SBERT encoding overhead" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Direct integration with model's internal representations" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Optimization 2: Reduced Action Generation Frequency" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Generates action every 5 steps instead of every step" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Assumes similar contexts produce similar optimal parameters" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "- Significant inference speedup with minimal accuracy loss" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Record start time
start_time=$(date +%s)

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-optimized-rl-test-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-optimized-rl \
    --use-stepwise-rl \
    --action-generation-freq 5 \
    --entropy-weight 0.1 \
    --online-lr 0.001 \
    --online-policy-save-path "log/${DATE}_optimized_rl_test/optimized/optimized_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 256 \
    --num-choices 1 \
    --checkpoint-dir "log/${DATE}_optimized_rl_test/optimized/checkpoints" \
    --checkpoint-freq 25 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    --answer-file "log/${DATE}_optimized_rl_test/optimized/answers.jsonl" \
    2>&1 | tee -a log/${DATE}_optimized_rl_test/optimized/output.log

# Record end time
end_time=$(date +%s)
optimized_duration=$((end_time - start_time))

echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "=== Phase 3: Performance Comparison ===" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Calculate speedup
if [ $standard_duration -gt 0 ]; then
    speedup=$(echo "scale=2; $standard_duration / $optimized_duration" | bc -l)
    improvement=$(echo "scale=1; ($standard_duration - $optimized_duration) * 100 / $standard_duration" | bc -l)
else
    speedup="N/A"
    improvement="N/A"
fi

echo "⏱️  Execution Time Comparison:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   Standard RL Policy:  ${standard_duration}s" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   Optimized RL Policy: ${optimized_duration}s" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   Speedup:             ${speedup}x" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   Time Improvement:    ${improvement}%" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Extract metrics from logs
echo "📊 Key Performance Metrics:" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Standard policy metrics
if [ -f "log/${DATE}_optimized_rl_test/standard/output.log" ]; then
    standard_actions=$(grep -c "Generated new action" log/${DATE}_optimized_rl_test/standard/output.log || echo "0")
    standard_cached=$(grep -c "Using cached action" log/${DATE}_optimized_rl_test/standard/output.log || echo "0")
    standard_total=$((standard_actions + standard_cached))
    
    echo "Standard RL Policy:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Action generations: $standard_actions" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Total steps:        $standard_total" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Action frequency:   Every step (100%)" | tee -a log/${DATE}_optimized_rl_test/comparison.log
fi

# Optimized policy metrics
if [ -f "log/${DATE}_optimized_rl_test/optimized/output.log" ]; then
    optimized_actions=$(grep -c "Generated new action" log/${DATE}_optimized_rl_test/optimized/output.log || echo "0")
    optimized_cached=$(grep -c "Using cached action" log/${DATE}_optimized_rl_test/optimized/output.log || echo "0")
    optimized_total=$((optimized_actions + optimized_cached))
    
    if [ $optimized_total -gt 0 ]; then
        action_reduction=$(echo "scale=1; $optimized_cached * 100 / $optimized_total" | bc -l)
    else
        action_reduction="N/A"
    fi
    
    echo "Optimized RL Policy:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Action generations: $optimized_actions" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Cached actions:     $optimized_cached" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Total steps:        $optimized_total" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "   Cache hit rate:     ${action_reduction}%" | tee -a log/${DATE}_optimized_rl_test/comparison.log
fi

echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "🔍 Technical Details:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Standard Policy State Representation:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - SBERT text embeddings (384 dimensions)" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - Requires text encoding at every step" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - External dependency on sentence transformers" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Optimized Policy State Representation:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - EAGLE-3 layer feature concatenation (~12,288 dimensions)" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - Direct use of model's internal hidden states" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - No external encoding overhead" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "   - Richer semantic information from multiple layers" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Check if wandb logs are available
echo "📈 Training Progress:" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "Check wandb dashboard for detailed training metrics and comparisons" | tee -a log/${DATE}_optimized_rl_test/comparison.log

# Quality analysis (basic check)
echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "📋 Output Quality Check:" | tee -a log/${DATE}_optimized_rl_test/comparison.log

if [ -f "log/${DATE}_optimized_rl_test/standard/answers.jsonl" ] && [ -f "log/${DATE}_optimized_rl_test/optimized/answers.jsonl" ]; then
    standard_lines=$(wc -l < "log/${DATE}_optimized_rl_test/standard/answers.jsonl")
    optimized_lines=$(wc -l < "log/${DATE}_optimized_rl_test/optimized/answers.jsonl")
    
    echo "Standard policy generated:  $standard_lines answers" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    echo "Optimized policy generated: $optimized_lines answers" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    
    if [ "$standard_lines" -eq "$optimized_lines" ]; then
        echo "✅ Both policies generated the same number of answers" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    else
        echo "⚠️  Different number of answers generated" | tee -a log/${DATE}_optimized_rl_test/comparison.log
    fi
else
    echo "⚠️  Answer files not found for comparison" | tee -a log/${DATE}_optimized_rl_test/comparison.log
fi

echo "" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "=== Test Complete ===" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "📁 Results saved in: log/${DATE}_optimized_rl_test/" | tee -a log/${DATE}_optimized_rl_test/comparison.log
echo "📊 Full comparison log: log/${DATE}_optimized_rl_test/comparison.log" | tee -a log/${DATE}_optimized_rl_test/comparison.log

echo ""
echo "=== Summary ==="
echo "✅ Optimized RL Policy Test Complete"
echo "🚀 Key Optimizations Tested:"
echo "   1. Layer Feature Concatenation (EAGLE-3 features vs SBERT)"
echo "   2. Reduced Action Generation Frequency (every 5 steps vs every step)"
echo "⏱️  Expected Benefits:"
echo "   - Faster inference due to reduced action generation overhead"
echo "   - No SBERT encoding cost"
echo "   - Richer state representation from multiple model layers"
echo "📁 Check results in: log/${DATE}_optimized_rl_test/"
