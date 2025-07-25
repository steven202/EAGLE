#!/bin/bash

# OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation
# This script demonstrates the optimized policy with:
# 1. EAGLE-3 layer features instead of SBERT text embeddings
# 2. Action caching to reduce computation frequency (every N steps)

DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_optimized_ppo"
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=200

# Create log directory
mkdir -p log/$DATE/{optimized_max_entropy_ppo,optimized_standard_ppo,baseline_results,evaluation}
mkdir -p log/$DATE/optimized_max_entropy_ppo/{checkpoints,evaluation}
mkdir -p log/$DATE/optimized_standard_ppo/{checkpoints,evaluation}

echo "=== OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 1: EAGLE-3 layer features instead of SBERT embeddings" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 2: Action caching - generate action every 10 steps" | tee -a log/$DATE/comparison.txt
echo "Expected speedup: ~50% reduction in RL policy computation" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Benchmark names for testing
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARK_NAMES=("GSM8K" "MT-Bench")

echo "=== Phase 1: Training with OPTIMIZED MAX-ENTROPY PPO ===" | tee -a log/$DATE/comparison.txt
echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
echo "- High entropy coefficient 0.1 for exploration" | tee -a log/$DATE/comparison.txt
echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id optimized_max_entropy_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $QUESTION_END \
    --answer-file log/$DATE/optimized_max_entropy_ppo/training_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory 80GiB \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-sb3-discrete-ppo \
    --online-lr 3e-4 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --enable-max-entropy \
    --max-entropy-ent-coef 0.1 \
    --inference-temperature 1.5 \
    --max-entropy-inference \
    --action-cache-steps 10 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --checkpoint-dir log/$DATE/optimized_max_entropy_ppo/checkpoints \
    --checkpoint-freq 50 \
    --wandb-project eagle-optimized-sb3-ppo \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo/training.log

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 2: Training with OPTIMIZED STANDARD PPO ===" | tee -a log/$DATE/comparison.txt
echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id optimized_standard_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $QUESTION_END \
    --answer-file log/$DATE/optimized_standard_ppo/training_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory 80GiB \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-sb3-discrete-ppo \
    --online-lr 3e-4 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --disable-max-entropy \
    --action-cache-steps 10 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --checkpoint-dir log/$DATE/optimized_standard_ppo/checkpoints \
    --checkpoint-freq 50 \
    --wandb-project eagle-optimized-sb3-ppo \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo/training.log

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Testing both optimized trained policies on ${#BENCHMARKS[@]} benchmarks:" | tee -a log/$DATE/comparison.txt
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    echo "$((i+1)). $benchmark_name - $benchmark" | tee -a log/$DATE/comparison.txt
done
echo "" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist
if [ ! -f "log/$DATE/optimized_max_entropy_ppo/optimized_max_entropy_ppo_policy_sb3.zip" ]; then
    echo "❌ Optimized max-entropy policy not found!" | tee -a log/$DATE/comparison.txt
    exit 1
fi

if [ ! -f "log/$DATE/optimized_standard_ppo/optimized_standard_ppo_policy_sb3.zip" ]; then
    echo "❌ Optimized standard policy not found!" | tee -a log/$DATE/comparison.txt
    exit 1
fi

echo "✅ Both optimized trained policies found. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Evaluate both policies on all benchmarks
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== Evaluating $benchmark_name with Optimized Max-Entropy PPO ===" | tee -a log/$DATE/comparison.txt
    
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --model-path $MODEL_PATH \
        --base-model-path $BASE_MODEL_PATH \
        --model-id optimized_max_entropy_ppo_$benchmark \
        --question-file eagle/data/$benchmark/question.jsonl \
        --question-begin 0 \
        --question-end -1 \
        --answer-file log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_answers.jsonl \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory 80GiB \
        --dtype float16 \
        --temperature 0.0 \
        --use-online-rl \
        --use-optimized-sb3-discrete-ppo \
        --online-inference-only \
        --online-policy-path log/$DATE/optimized_max_entropy_ppo/optimized_max_entropy_ppo_policy_sb3.zip \
        --enable-max-entropy \
        --inference-temperature 1.5 \
        --max-entropy-inference \
        --action-cache-steps 10 \
        --action-cache-enabled \
        --use-eagle3-features \
        --hidden-size 4096 \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_evaluation.log
    
    echo "=== Evaluating $benchmark_name with Optimized Standard PPO ===" | tee -a log/$DATE/comparison.txt
    
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --model-path $MODEL_PATH \
        --base-model-path $BASE_MODEL_PATH \
        --model-id optimized_standard_ppo_$benchmark \
        --question-file eagle/data/$benchmark/question.jsonl \
        --question-begin 0 \
        --question-end -1 \
        --answer-file log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_answers.jsonl \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory 80GiB \
        --dtype float16 \
        --temperature 0.0 \
        --use-online-rl \
        --use-optimized-sb3-discrete-ppo \
        --online-inference-only \
        --online-policy-path log/$DATE/optimized_standard_ppo/optimized_standard_ppo_policy_sb3.zip \
        --disable-max-entropy \
        --action-cache-steps 10 \
        --action-cache-enabled \
        --use-eagle3-features \
        --hidden-size 4096 \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_evaluation.log
done

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 4: Performance Analysis & Baseline Comparison ===" | tee -a log/$DATE/comparison.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Generate baseline results for all benchmarks (LLaMA 3.1 8B)
echo "=== Generating LLaMA 3.1 8B Baseline Results ===" | tee -a log/$DATE/comparison.txt

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Baseline evaluation on $benchmark..." | tee -a log/$DATE/comparison.txt
    
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --model-path $MODEL_PATH \
        --base-model-path $BASE_MODEL_PATH \
        --model-id baseline_llama31_8b_$benchmark \
        --question-file eagle/data/$benchmark/question.jsonl \
        --question-begin 0 \
        --question-end -1 \
        --answer-file log/$DATE/baseline_results/${benchmark}_baseline_answers.jsonl \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory 80GiB \
        --dtype float16 \
        --temperature 0.0 \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-eagle3 2>&1 | tee log/$DATE/baseline_results/${benchmark}_baseline.log
done

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Summary ===" | tee -a log/$DATE/comparison.txt
echo "Training completed with optimized policies!" | tee -a log/$DATE/comparison.txt
echo "Results saved in: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "Key optimizations implemented:" | tee -a log/$DATE/comparison.txt
echo "1. EAGLE-3 layer features instead of SBERT text embeddings" | tee -a log/$DATE/comparison.txt
echo "2. Action caching every 10 steps (~50% computation reduction)" | tee -a log/$DATE/comparison.txt
echo "3. Maintained compatibility with existing max-entropy and standard modes" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create performance summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Date: $DATE" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Optimizations: EAGLE-3 features + Action caching" >> log/$DATE/summary.txt
echo "Training questions: $QUESTION_END" >> log/$DATE/summary.txt
echo "Benchmarks evaluated: ${BENCHMARKS[*]}" >> log/$DATE/summary.txt
echo "" >> log/$DATE/summary.txt

echo "🎉 Optimized EAGLE RL training and evaluation completed successfully!" | tee -a log/$DATE/comparison.txt
echo "Check log/$DATE/summary.txt for detailed results." | tee -a log/$DATE/comparison.txt
