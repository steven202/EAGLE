#!/bin/bash

# OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation
# This script demonstrates the optimized policy with:
# 1. EAGLE-3 layer features instead of SBERT text embeddings
# 2. Action caching to reduce computation frequency (every N steps)
# 3. NEW: Context-only state representation option (SBERT embeddings directly)
#  --use-context-only-state \
DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_optimized_ppo"
DATE='20250724_2135_optimized_ppo_30'
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=200

# Create log directory
mkdir -p log/$DATE/{optimized_max_entropy_ppo,optimized_standard_ppo,baseline_results}
mkdir -p log/$DATE/optimized_max_entropy_ppo/{checkpoints,evaluation}
mkdir -p log/$DATE/optimized_standard_ppo/{checkpoints,evaluation}

echo "=== OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 1: EAGLE-3 layer features instead of SBERT embeddings" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 2: Action caching - generate action every 10 steps" | tee -a log/$DATE/comparison.txt
# echo "OPTIMIZATION 3: Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt --use-context-only-state \
echo "Expected speedup: ~50% reduction in RL policy computation" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARK_NAMES=("GSM8K" "MT-Bench")
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARKS=("gsm8k" "mt_bench")

echo "=== Phase 1: Training with OPTIMIZED MAX-ENTROPY PPO ===" | tee -a log/$DATE/comparison.txt
echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
echo "- High entropy coefficient 0.1 for exploration" | tee -a log/$DATE/comparison.txt
echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --ea-model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id optimized_max_entropy_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $QUESTION_END \
    --answer-file log/$DATE/optimized_max_entropy_ppo/training_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
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
    --action-cache-steps 30 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --use-context-only-state \
    --checkpoint-dir log/$DATE/optimized_max_entropy_ppo/checkpoints \
    --online-policy-save-path log/$DATE/optimized_max_entropy_ppo/optimized_max_entropy_ppo_policy_sb3.zip \
    --checkpoint-freq 500 \
    --wandb-project eagle-optimized-sb3-ppo \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-stepwise-rl \
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
    --ea-model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id optimized_standard_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $QUESTION_END \
    --answer-file log/$DATE/optimized_standard_ppo/training_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-sb3-discrete-ppo \
    --online-lr 3e-4 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --disable-max-entropy \
    --action-cache-steps 30 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --use-context-only-state \
    --checkpoint-dir log/$DATE/optimized_standard_ppo/checkpoints \
    --online-policy-save-path log/$DATE/optimized_standard_ppo/optimized_standard_ppo_policy_sb3.zip \
    --checkpoint-freq 500 \
    --wandb-project eagle-optimized-sb3-ppo \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-stepwise-rl \
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

    if [ ! -f log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_results.jsonl ]; then
        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_max_entropy_ppo_$benchmark \
            --question-file eagle/data/$benchmark/question.jsonl \
            --question-begin 0 \
            --question-end -1 \
            --answer-file log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_results.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-sb3-discrete-ppo \
            --online-inference-only \
            --use-context-only-state \
            --online-policy-path log/$DATE/optimized_max_entropy_ppo/optimized_max_entropy_ppo_policy_sb3.zip \
            --enable-max-entropy \
            --inference-temperature 1.5 \
            --max-entropy-inference \
            --action-cache-steps 30 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_evaluation.log
    fi
    
    echo "=== Evaluating $benchmark_name with Optimized Standard PPO ===" | tee -a log/$DATE/comparison.txt
    
    if [ ! -f log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_results.jsonl ]; then
        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_standard_ppo_$benchmark \
            --question-file eagle/data/$benchmark/question.jsonl \
            --question-begin 0 \
            --question-end -1 \
            --answer-file log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_results.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-sb3-discrete-ppo \
            --online-inference-only \
            --use-context-only-state \
            --online-policy-path log/$DATE/optimized_standard_ppo/optimized_standard_ppo_policy_sb3.zip \
            --disable-max-entropy \
            --action-cache-steps 30 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_evaluation.log
    fi
done


echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 4: Performance Analysis Across All Benchmarks ===" | tee -a log/$DATE/comparison.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create consolidated results summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Training Date: $DATE" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Base Model: $BASE_MODEL_PATH" >> log/$DATE/summary.txt
echo "" >> log/$DATE/summary.txt

# Generate baseline results for all benchmarks (LLaMA 3.1 8B)
echo "=== Generating LLaMA 3.1 8B Baseline Results ===" | tee -a log/$DATE/comparison.txt

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Generating baseline for $benchmark..." | tee -a log/$DATE/comparison.txt
    
    # Generate EAGLE3 baseline
    if [ ! -f "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" ]; then
        python -m eagle.evaluation.gen_ea_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" \
            --temperature 0.0 \
            --use_eagle3 \
            2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_eagle3.log
    else
        echo "EAGLE3 baseline for $benchmark already exists" | tee -a log/$DATE/comparison.txt
    fi
    
    # Generate standard baseline
    if [ ! -f "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" ]; then
        python -m eagle.evaluation.gen_baseline_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" \
            --temperature 0.0 \
            2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_standard.log
    else
        echo "Standard baseline for $benchmark already exists" | tee -a log/$DATE/comparison.txt
    fi
done

# Analyze results for each benchmark with comprehensive comparisons
echo "" | tee -a log/$DATE/comparison.txt
echo "=== Comprehensive Performance Analysis ===" | tee -a log/$DATE/comparison.txt

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== $benchmark_name - $benchmark Performance Analysis ===" | tee -a log/$DATE/comparison.txt
    echo "Benchmark: $benchmark_name - $benchmark" >> log/$DATE/summary.txt
    echo "===========================================" >> log/$DATE/summary.txt
    
    # Define result files
    max_file="log/$DATE/optimized_max_entropy_ppo/evaluation/${benchmark}_results.jsonl"
    std_file="log/$DATE/optimized_standard_ppo/evaluation/${benchmark}_results.jsonl"
    eagle3_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl"
    baseline_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl"
    
    if [ -f "$max_file" ] && [ -f "$std_file" ] && [ -f "$eagle3_file" ] && [ -f "$baseline_file" ]; then
        echo "✅ All result files found for $benchmark_name" | tee -a log/$DATE/comparison.txt
        
        # Speed comparison using existing speed.py tool (3 comprehensive comparisons like test_expanded_action_space.sh)
        if [ -f "eagle/evaluation/speed.py" ]; then
            echo "" >> log/$DATE/summary.txt
            echo "1. Max-Entropy PPO vs EAGLE3 Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "2. Max-Entropy PPO vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "3. EAGLE3 Baseline vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$eagle3_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "4. Standard PPO vs EAGLE3 Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "5. Standard PPO vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "6. Max-Entropy PPO vs Standard PPO:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$std_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
                
        else
            echo "Speed analysis tool not found" >> log/$DATE/summary.txt
        fi
        
        # Basic statistics
        echo "" >> log/$DATE/summary.txt
        echo "Result File Statistics:" >> log/$DATE/summary.txt
        echo "Max-Entropy PPO: $(wc -l < "$max_file") samples" >> log/$DATE/summary.txt
        echo "Standard PPO: $(wc -l < "$std_file") samples" >> log/$DATE/summary.txt
        echo "EAGLE3 Baseline: $(wc -l < "$eagle3_file") samples" >> log/$DATE/summary.txt
        echo "Standard Baseline: $(wc -l < "$baseline_file") samples" >> log/$DATE/summary.txt
        
    else
        echo "❌ Missing result files for $benchmark_name" | tee -a log/$DATE/comparison.txt
        echo "Required files:" | tee -a log/$DATE/comparison.txt
        echo "  Max-Entropy PPO: $max_file $([ -f "$max_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  Standard PPO: $std_file $([ -f "$std_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  EAGLE3 Baseline: $eagle3_file $([ -f "$eagle3_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  Standard Baseline: $baseline_file $([ -f "$baseline_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "Missing result files - check evaluation and baseline generation logs" >> log/$DATE/summary.txt
    fi
    
    echo "" >> log/$DATE/summary.txt
    echo "" | tee -a log/$DATE/comparison.txt
done

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Summary ===" | tee -a log/$DATE/comparison.txt
echo "Training completed with optimized PPO policies!" | tee -a log/$DATE/comparison.txt
echo "Results saved in: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "Key optimizations implemented:" | tee -a log/$DATE/comparison.txt
echo "1. EAGLE-3 layer features instead of SBERT text embeddings" | tee -a log/$DATE/comparison.txt
echo "2. Action caching every 10 steps (~50% computation reduction)" | tee -a log/$DATE/comparison.txt
echo "3. Maintained compatibility with existing max-entropy and standard modes" | tee -a log/$DATE/comparison.txt
echo "4. Enhanced PPO with temperature-based action sampling" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create performance summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Date: $DATE" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Algorithm: Optimized PPO" >> log/$DATE/summary.txt
echo "Optimizations: EAGLE-3 features + Action caching" >> log/$DATE/summary.txt
echo "Training questions: $QUESTION_END" >> log/$DATE/summary.txt
echo "Benchmarks evaluated: ${BENCHMARKS[*]}" >> log/$DATE/summary.txt
echo "" >> log/$DATE/summary.txt

# echo "🎉 Optimized EAGLE PPO training and evaluation completed successfully!" | tee -a log/$DATE/comparison.txt
echo "Check log/$DATE/summary.txt for detailed results." | tee -a log/$DATE/comparison.txt

echo "file location: log/$DATE/summary.txt"