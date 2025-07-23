#!/bin/bash

# EAGLE SB3 PPO Formal Training & Multi-Benchmark Evaluation: Standard vs Max-Entropy
# This script performs formal training with both PPO modes and comprehensive testing
# NOTE: Max-Entropy PPO is now the DEFAULT mode. Use --disable-max-entropy for Standard PPO.

DATE=$(date '+%Y%m%d_%H%M')
DATE='20250721_1215'
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=8600

# Create single unified log directory
mkdir -p log/$DATE/{standard_ppo,max_entropy_ppo,baseline_results,evaluation}
mkdir -p log/$DATE/standard_ppo/{checkpoints,evaluation}
mkdir -p log/$DATE/max_entropy_ppo/{checkpoints,evaluation}

echo "=== EAGLE SB3 PPO Formal Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "Evaluation Benchmarks: MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail" | tee -a log/$DATE/comparison.txt
echo "Auto-Resume: Enabled - resumes from checkpoints if available" | tee -a log/$DATE/comparison.txt
echo "Default Mode: Max-Entropy PPO - use --disable-max-entropy for Standard PPO" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Benchmark names for testing
# BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
# BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARK_NAMES=("GSM8K" "MT-Bench")
# BENCHMARKS=("gsm8k")
# BENCHMARK_NAMES=("GSM8K")

echo "=== Phase 1: Training with MAX-ENTROPY PPO - Default Mode ===" | tee -a log/$DATE/comparison.txt
echo "- High entropy coefficient 0.1" | tee -a log/$DATE/comparison.txt
echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
echo "- Enhanced exploration during training and inference" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: eagle/data/rl_training/question.jsonl - questions 0-10000 for faster training" | tee -a log/$DATE/comparison.txt
echo "- Auto-resume: Enabled" | tee -a log/$DATE/comparison.txt
echo "- NOTE: Max-entropy mode is now the default - no explicit flags needed" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-max-entropy-ppo-formal-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --use-stepwise-rl \
    --online-lr 0.0003 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --ppo-clip-range 0.2 \
    --ppo-gamma 0.95 \
    --ppo-gae-lambda 0.9 \
    --ppo-vf-coef 0.5 \
    --online-policy-save-path "log/$DATE/max_entropy_ppo/max_entropy_ppo_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE/max_entropy_ppo/checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    2>&1 | tee -a log/$DATE/max_entropy_ppo/training.log

# echo "" | tee -a log/$DATE/comparison.txt
# echo "=== Phase 2: Training with STANDARD PPO - Non-Default Mode ===" | tee -a log/$DATE/comparison.txt
# echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
# echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
# echo "- Standard exploration during training" | tee -a log/$DATE/comparison.txt
# echo "- Training dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
# echo "- Auto-resume: Enabled" | tee -a log/$DATE/comparison.txt
# echo "- NOTE: Using --disable-max-entropy to override default max-entropy mode" | tee -a log/$DATE/comparison.txt
# echo "" | tee -a log/$DATE/comparison.txt

# PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-standard-ppo-formal-$DATE" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --use-sb3-discrete-ppo \
#     --disable-max-entropy \
#     --online-lr 0.0003 \
#     --ppo-n-steps 64 \
#     --ppo-batch-size 32 \
#     --ppo-epochs 4 \
#     --ppo-clip-range 0.2 \
#     --ppo-gamma 0.99 \
#     --ppo-gae-lambda 0.95 \
#     --ppo-ent-coef 0.01 \
#     --ppo-vf-coef 0.5 \
#     --online-policy-save-path "log/$DATE/standard_ppo/standard_ppo_policy.pth" \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --checkpoint-dir "log/$DATE/standard_ppo/checkpoints" \
#     --checkpoint-freq 50 \
#     --online-repeat-factor 1 \
#     --question-file eagle/data/rl_training/question.jsonl \
#     --question-begin 0 --question-end $QUESTION_END \
#     2>&1 | tee -a log/$DATE/standard_ppo/training.log

# echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Testing both trained policies on 5 benchmarks:" | tee -a log/$DATE/comparison.txt
echo "1. MT-Bench - multi-turn conversation" | tee -a log/$DATE/comparison.txt
echo "2. HumanEval - code generation" | tee -a log/$DATE/comparison.txt
echo "3. GSM8K - mathematical reasoning" | tee -a log/$DATE/comparison.txt
echo "4. Alpaca - instruction following" | tee -a log/$DATE/comparison.txt
echo "5. CNN/DailyMail - text summarization" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist
if [ ! -f "log/$DATE/max_entropy_ppo/max_entropy_ppo_policy_sb3.zip" ]; then
    echo "❌ Error: Max-entropy PPO policy not found!" | tee -a log/$DATE/comparison.txt
    echo "   Expected: log/$DATE/max_entropy_ppo/max_entropy_ppo_policy_sb3.zip" | tee -a log/$DATE/comparison.txt
    exit 1
fi

if [ ! -f "log/$DATE/standard_ppo/standard_ppo_policy_sb3.zip" ]; then
    echo "❌ Error: Standard PPO policy not found!" | tee -a log/$DATE/comparison.txt
    echo "   Expected: log/$DATE/standard_ppo/standard_ppo_policy_sb3.zip" | tee -a log/$DATE/comparison.txt
    exit 1
fi

echo "✅ Both trained policies found. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Evaluate both policies on all benchmarks
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== Evaluating on $benchmark_name - $benchmark ===" | tee -a log/$DATE/comparison.txt
    
    # Max-Entropy PPO Evaluation (now tested first - uses default behavior)
    echo "Testing Max-Entropy PPO on $benchmark_name..." | tee -a log/$DATE/comparison.txt
    python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --base-model-path "$BASE_MODEL_PATH" \
        --ea-model-path "$MODEL_PATH" \
        --model-id "eagle-max-entropy-ppo-$benchmark-$DATE" \
        --bench-name "$benchmark" \
        --use-online-rl \
        --use-sb3-discrete-ppo \
        --online-policy-path "log/$DATE/max_entropy_ppo/max_entropy_ppo_policy.pth" \
        --online-inference-only \
        --temperature 0.0 \
        --use_eagle3 \
        --max-new-token 512 \
        --num-choices 1 \
        --answer-file "log/$DATE/max_entropy_ppo/evaluation/${benchmark}_results.jsonl" \
        --no-wandb \
        2>&1 | tee -a log/$DATE/max_entropy_ppo/evaluation/${benchmark}_test.log
    
    # Standard PPO Evaluation (now tested second)
    # echo "Testing Standard PPO on $benchmark_name..." | tee -a log/$DATE/comparison.txt
    # python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    #     --base-model-path "$BASE_MODEL_PATH" \
    #     --ea-model-path "$MODEL_PATH" \
    #     --model-id "eagle-standard-ppo-$benchmark-$DATE" \
    #     --bench-name "$benchmark" \
    #     --use-online-rl \
    #     --use-sb3-discrete-ppo \
    #     --disable-max-entropy \
    #     --online-policy-path "log/$DATE/standard_ppo/standard_ppo_policy.pth" \
    #     --online-inference-only \
    #     --temperature 0.0 \
    #     --use_eagle3 \
    #     --max-new-token 512 \
    #     --num-choices 1 \
    #     --answer-file "log/$DATE/standard_ppo/evaluation/${benchmark}_results.jsonl" \
    #     --no-wandb \
    #     2>&1 | tee -a log/$DATE/standard_ppo/evaluation/${benchmark}_test.log
    
    echo "✅ Completed evaluation on $benchmark_name" | tee -a log/$DATE/comparison.txt
    echo "" | tee -a log/$DATE/comparison.txt
done

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 4: Performance Analysis Across All Benchmarks ===" | tee -a log/$DATE/comparison.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create consolidated results summary
echo "Performance Summary Report" > log/$DATE/summary.txt
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
            2>&1 | tee -a log/$DATE/baseline_${benchmark}_eagle3.log
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
            2>&1 | tee -a log/$DATE/baseline_${benchmark}_standard.log
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
    max_file="log/$DATE/max_entropy_ppo/evaluation/${benchmark}_results.jsonl"
    std_file="log/$DATE/standard_ppo/evaluation/${benchmark}_results.jsonl"
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

