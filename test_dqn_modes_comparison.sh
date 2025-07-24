#!/bin/bash

# EAGLE DQN Formal Training & Multi-Benchmark Evaluation: Standard vs Max-Entropy
# This script performs formal training with both DQN modes and comprehensive testing
# NOTE: Max-Entropy DQN is now the DEFAULT mode. Use --disable-max-entropy for Standard DQN.

DATE=$(date '+%Y%m%d_%H%M')
DATE='20250724_0357'
DATE="${DATE}_dqn"
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=200

# Create single unified log directory
mkdir -p log/$DATE/{standard_dqn,max_entropy_dqn,baseline_results,evaluation}
mkdir -p log/$DATE/standard_dqn/{checkpoints,evaluation}
mkdir -p log/$DATE/max_entropy_dqn/{checkpoints,evaluation}

echo "=== EAGLE DQN Formal Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "Evaluation Benchmarks: MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail" | tee -a log/$DATE/comparison.txt
echo "Auto-Resume: Enabled - resumes from checkpoints if available" | tee -a log/$DATE/comparison.txt
echo "Default Mode: Max-Entropy DQN - use --disable-max-entropy for Standard DQN" | tee -a log/$DATE/comparison.txt
echo "Action Space: 5×5×5 = 125 total (116 valid with constraints)" | tee -a log/$DATE/comparison.txt
echo "Constraint: total_tokens ≤ top_k^(depth-1)" | tee -a log/$DATE/comparison.txt
echo "Stepwise Action Generation: Enabled for dynamic parameter optimization" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Benchmark names for testing
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARK_NAMES=("GSM8K" "MT-Bench")
# Alternative: Full benchmark suite
# BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum")
# BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail")

echo "=== Phase 1: Training with MAX-ENTROPY DQN - Default Mode ===" | tee -a log/$DATE/comparison.txt
echo "- High entropy coefficient and temperature-based sampling" | tee -a log/$DATE/comparison.txt
echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
echo "- Enhanced exploration during training and inference" | tee -a log/$DATE/comparison.txt
echo "- Stepwise action generation enabled" | tee -a log/$DATE/comparison.txt
echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: eagle/data/rl_training/question.jsonl - questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
echo "- Auto-resume: Enabled" | tee -a log/$DATE/comparison.txt
echo "- NOTE: Max-entropy mode is now the default - no explicit flags needed" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-max-entropy-dqn-formal-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-stepwise-rl \
    --online-lr 0.01 \
    --online-epsilon-start 0.9 \
    --online-epsilon-end 0.3 \
    --online-memory-size 100 \
    --online-batch-size 8 \
    --online-policy-save-path "log/$DATE/max_entropy_dqn/max_entropy_dqn_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE/max_entropy_dqn/checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    2>&1 | tee -a log/$DATE/max_entropy_dqn/training.log

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 2: Training with STANDARD DQN - Non-Default Mode ===" | tee -a log/$DATE/comparison.txt
echo "- Low temperature and deterministic inference" | tee -a log/$DATE/comparison.txt
echo "- Standard epsilon-greedy exploration" | tee -a log/$DATE/comparison.txt
echo "- Stepwise action generation enabled" | tee -a log/$DATE/comparison.txt
echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
echo "- Training dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "- Auto-resume: Enabled" | tee -a log/$DATE/comparison.txt
echo "- NOTE: Using --disable-max-entropy to override default max-entropy mode" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-standard-dqn-formal-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-stepwise-rl \
    --disable-max-entropy \
    --online-lr 0.01 \
    --online-epsilon-start 0.9 \
    --online-epsilon-end 0.3 \
    --online-memory-size 100 \
    --online-batch-size 8 \
    --online-policy-save-path "log/$DATE/standard_dqn/standard_dqn_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/$DATE/standard_dqn/checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    2>&1 | tee -a log/$DATE/standard_dqn/training.log

# echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Testing both trained policies on ${#BENCHMARKS[@]} benchmarks with stepwise action generation:" | tee -a log/$DATE/comparison.txt
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    echo "$((i+1)). $benchmark_name - $benchmark" | tee -a log/$DATE/comparison.txt
done
echo "Stepwise Mode: Dynamic parameter optimization per inference step" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist
if [ ! -f "log/$DATE/max_entropy_dqn/max_entropy_dqn_policy.pth" ]; then
    echo "❌ Error: Max-entropy DQN policy not found!" | tee -a log/$DATE/comparison.txt
    echo "   Expected: log/$DATE/max_entropy_dqn/max_entropy_dqn_policy.pth" | tee -a log/$DATE/comparison.txt
    exit 1
fi

if [ ! -f "log/$DATE/standard_dqn/standard_dqn_policy.pth" ]; then
    echo "❌ Error: Standard DQN policy not found!" | tee -a log/$DATE/comparison.txt
    echo "   Expected: log/$DATE/standard_dqn/standard_dqn_policy.pth" | tee -a log/$DATE/comparison.txt
    exit 1
fi

echo "✅ Both trained policies found. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Evaluate both policies on all benchmarks
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== Evaluating on $benchmark_name - $benchmark ===" | tee -a log/$DATE/comparison.txt
    
    # Max-Entropy DQN Evaluation (now tested first - uses default behavior)
    # check if the evaluation file already exists, if not, run the evaluation
    if [ ! -f "log/$DATE/max_entropy_dqn/evaluation/${benchmark}_results.jsonl" ]; then
        echo "Testing Max-Entropy DQN on $benchmark_name..." | tee -a log/$DATE/comparison.txt
        python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --base-model-path "$BASE_MODEL_PATH" \
            --ea-model-path "$MODEL_PATH" \
            --model-id "eagle-max-entropy-dqn-$benchmark-$DATE" \
            --bench-name "$benchmark" \
            --use-online-rl \
            --use-stepwise-rl \
            --online-policy-path "log/$DATE/max_entropy_dqn/max_entropy_dqn_policy.pth" \
            --online-inference-only \
            --temperature 0.0 \
            --use_eagle3 \
            --num-choices 1 \
            --answer-file "log/$DATE/max_entropy_dqn/evaluation/${benchmark}_results.jsonl" \
            --no-wandb \
            2>&1 | tee -a log/$DATE/max_entropy_dqn/evaluation/${benchmark}_test.log
    else
        echo "Max-Entropy DQN evaluation for $benchmark_name already exists, skipping..." | tee -a log/$DATE/comparison.txt
    fi
    
    if [ ! -f "log/$DATE/standard_dqn/evaluation/${benchmark}_results.jsonl" ]; then
        # Standard DQN Evaluation (now tested second)
        echo "Testing Standard DQN on $benchmark_name..." | tee -a log/$DATE/comparison.txt
        python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --base-model-path "$BASE_MODEL_PATH" \
            --ea-model-path "$MODEL_PATH" \
            --model-id "eagle-standard-dqn-$benchmark-$DATE" \
            --bench-name "$benchmark" \
            --use-online-rl \
            --use-stepwise-rl \
            --disable-max-entropy \
            --online-policy-path "log/$DATE/standard_dqn/standard_dqn_policy.pth" \
            --online-inference-only \
            --temperature 0.0 \
            --use_eagle3 \
            --max-new-token 512 \
            --num-choices 1 \
            --answer-file "log/$DATE/standard_dqn/evaluation/${benchmark}_results.jsonl" \
            --no-wandb \
            2>&1 | tee -a log/$DATE/standard_dqn/evaluation/${benchmark}_test.log
    else
        echo "Standard DQN evaluation for $benchmark_name already exists, skipping..." | tee -a log/$DATE/comparison.txt
    fi
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
echo "Algorithm: DQN (Deep Q-Network)" >> log/$DATE/summary.txt
echo "Action Space: 5×5×5 = 125 total (116 valid with constraints)" >> log/$DATE/summary.txt
echo "Constraint: total_tokens ≤ top_k^(depth-1)" >> log/$DATE/summary.txt
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
    max_file="log/$DATE/max_entropy_dqn/evaluation/${benchmark}_results.jsonl"
    std_file="log/$DATE/standard_dqn/evaluation/${benchmark}_results.jsonl"
    eagle3_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl"
    baseline_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl"
    
    if [ -f "$max_file" ] && [ -f "$std_file" ] && [ -f "$eagle3_file" ] && [ -f "$baseline_file" ]; then
        echo "✅ All result files found for $benchmark_name" | tee -a log/$DATE/comparison.txt
        
        # Speed comparison using existing speed.py tool (comprehensive comparisons like test_expanded_action_space.sh)
        if [ -f "eagle/evaluation/speed.py" ]; then
            echo "" >> log/$DATE/summary.txt
            echo "1. Max-Entropy DQN vs EAGLE3 Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "2. Max-Entropy DQN vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
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
            echo "4. Standard DQN vs EAGLE3 Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "5. Standard DQN vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            echo "" >> log/$DATE/summary.txt
            echo "6. Max-Entropy DQN vs Standard DQN:" >> log/$DATE/summary.txt
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
        echo "Max-Entropy DQN: $(wc -l < "$max_file") samples" >> log/$DATE/summary.txt
        echo "Standard DQN: $(wc -l < "$std_file") samples" >> log/$DATE/summary.txt
        echo "EAGLE3 Baseline: $(wc -l < "$eagle3_file") samples" >> log/$DATE/summary.txt
        echo "Standard Baseline: $(wc -l < "$baseline_file") samples" >> log/$DATE/summary.txt
        
    else
        echo "❌ Missing result files for $benchmark_name" | tee -a log/$DATE/comparison.txt
        echo "Required files:" | tee -a log/$DATE/comparison.txt
        echo "  Max-Entropy DQN: $max_file $([ -f "$max_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  Standard DQN: $std_file $([ -f "$std_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  EAGLE3 Baseline: $eagle3_file $([ -f "$eagle3_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  Standard Baseline: $baseline_file $([ -f "$baseline_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "Missing result files - check evaluation and baseline generation logs" >> log/$DATE/summary.txt
    fi
    
    echo "" >> log/$DATE/summary.txt
    echo "" | tee -a log/$DATE/comparison.txt
done

echo "=== Analysis Complete ===" | tee -a log/$DATE/comparison.txt
echo "All results and logs saved to: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

echo "File Structure Summary:" | tee -a log/$DATE/comparison.txt
echo "- log/$DATE/comparison.txt - This comparison log" | tee -a log/$DATE/comparison.txt
echo "- log/$DATE/summary.txt - Performance analysis summary" | tee -a log/$DATE/comparison.txt
echo "- log/$DATE/max_entropy_dqn/ - Max-Entropy DQN training and evaluation results" | tee -a log/$DATE/comparison.txt
echo "- log/$DATE/standard_dqn/ - Standard DQN training and evaluation results" | tee -a log/$DATE/comparison.txt
echo "- log/$DATE/baseline_results/ - EAGLE3 and LLaMA baselines for comparison" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

echo "🎯 DQN vs PPO Comparison Notes:" | tee -a log/$DATE/comparison.txt
echo "- DQN: Uses Q-learning with experience replay and epsilon-greedy exploration" | tee -a log/$DATE/comparison.txt
echo "- PPO: Uses policy gradients with clipped surrogate objective" | tee -a log/$DATE/comparison.txt
echo "- DQN Settings: lr=0.01, ε=0.9→0.3, memory=100, batch=8 (from test_expanded_action_space.sh)" | tee -a log/$DATE/comparison.txt
echo "- PPO Settings: lr=0.0003, n_steps=64, batch=32, epochs=4 (from test_ppo_modes_comparison2.sh)" | tee -a log/$DATE/comparison.txt
echo "- Both support Max-Entropy modes for enhanced exploration" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

echo "✅ DQN Modes Comparison Complete!" | tee -a log/$DATE/comparison.txt
echo "Check log/$DATE/summary.txt for detailed performance analysis" | tee -a log/$DATE/comparison.txt
