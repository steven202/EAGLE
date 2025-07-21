#!/bin/bash

# EAGLE SB3 PPO Formal Training & Multi-Benchmark Evaluation: Standard vs Max-Entropy
# This script performs formal training with both PPO modes and comprehensive testing
# NOTE: Max-Entropy PPO is now the DEFAULT mode. Use --disable-max-entropy for Standard PPO.

DATE=$(date '+%Y%m%d_%H%M')
MODEL_PATH="/home/guo/EAGLE_RL/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=1000
echo "=== EAGLE SB3 PPO Formal Training & Multi-Benchmark Evaluation ===" | tee log_comparison_$DATE.txt
echo "Model: $MODEL_PATH" | tee -a log_comparison_$DATE.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log_comparison_$DATE.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log_comparison_$DATE.txt
echo "Evaluation Benchmarks: MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail" | tee -a log_comparison_$DATE.txt
echo "Auto-Resume: Enabled (resumes from checkpoints if available)" | tee -a log_comparison_$DATE.txt
echo "Default Mode: Max-Entropy PPO (use --disable-max-entropy for Standard PPO)" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

# Create directories
mkdir -p log/standard_ppo_$DATE/{checkpoints,evaluation}
mkdir -p log/max_entropy_ppo_$DATE/{checkpoints,evaluation}

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")

echo "=== Phase 1: Training with MAX-ENTROPY PPO (Default Mode) ===" | tee -a log_comparison_$DATE.txt
echo "- High entropy coefficient (0.1)" | tee -a log_comparison_$DATE.txt
echo "- Temperature-based inference (T=1.5)" | tee -a log_comparison_$DATE.txt
echo "- Enhanced exploration during training and inference" | tee -a log_comparison_$DATE.txt
echo "- Training dataset: eagle/data/rl_training/question.jsonl (questions 0-10000 for faster training)" | tee -a log_comparison_$DATE.txt
echo "- Auto-resume: Enabled" | tee -a log_comparison_$DATE.txt
echo "- NOTE: Max-entropy mode is now the default (no explicit flags needed)" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-max-entropy-ppo-formal-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --online-lr 0.0003 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --ppo-clip-range 0.2 \
    --ppo-gamma 0.95 \
    --ppo-gae-lambda 0.9 \
    --ppo-vf-coef 0.5 \
    --online-policy-save-path "log/max_entropy_ppo_$DATE/max_entropy_ppo_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/max_entropy_ppo_$DATE/checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    2>&1 | tee log/max_entropy_ppo_$DATE/training.log

echo "" | tee -a log_comparison_$DATE.txt
echo "=== Phase 2: Training with STANDARD PPO (Non-Default Mode) ===" | tee -a log_comparison_$DATE.txt
echo "- Low entropy coefficient (0.01)" | tee -a log_comparison_$DATE.txt
echo "- Deterministic inference" | tee -a log_comparison_$DATE.txt
echo "- Standard exploration during training" | tee -a log_comparison_$DATE.txt
echo "- Training dataset: eagle/data/rl_training/question.jsonl" | tee -a log_comparison_$DATE.txt
echo "- Auto-resume: Enabled" | tee -a log_comparison_$DATE.txt
echo "- NOTE: Using --disable-max-entropy to override default max-entropy mode" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$MODEL_PATH" \
    --model-id "eagle-standard-ppo-formal-$DATE" \
    --bench-name "mt_bench" \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --disable-max-entropy \
    --online-lr 0.0003 \
    --ppo-n-steps 64 \
    --ppo-batch-size 32 \
    --ppo-epochs 4 \
    --ppo-clip-range 0.2 \
    --ppo-gamma 0.99 \
    --ppo-gae-lambda 0.95 \
    --ppo-ent-coef 0.01 \
    --ppo-vf-coef 0.5 \
    --online-policy-save-path "log/standard_ppo_$DATE/standard_ppo_policy.pth" \
    --temperature 0.0 \
    --use_eagle3 \
    --max-new-token 512 \
    --num-choices 1 \
    --checkpoint-dir "log/standard_ppo_$DATE/checkpoints" \
    --checkpoint-freq 50 \
    --online-repeat-factor 1 \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 --question-end $QUESTION_END \
    2>&1 | tee log/standard_ppo_$DATE/training.log

echo "" | tee -a log_comparison_$DATE.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log_comparison_$DATE.txt
echo "Testing both trained policies on 5 benchmarks:" | tee -a log_comparison_$DATE.txt
echo "1. MT-Bench (multi-turn conversation)" | tee -a log_comparison_$DATE.txt
echo "2. HumanEval (code generation)" | tee -a log_comparison_$DATE.txt
echo "3. GSM8K (mathematical reasoning)" | tee -a log_comparison_$DATE.txt
echo "4. Alpaca (instruction following)" | tee -a log_comparison_$DATE.txt
echo "5. CNN/DailyMail (text summarization)" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

# Check if trained policies exist
if [ ! -f "log/max_entropy_ppo_$DATE/max_entropy_ppo_policy_sb3.zip" ]; then
    echo "❌ Error: Max-entropy PPO policy not found!" | tee -a log_comparison_$DATE.txt
    echo "   Expected: log/max_entropy_ppo_$DATE/max_entropy_ppo_policy_sb3.zip" | tee -a log_comparison_$DATE.txt
    exit 1
fi

if [ ! -f "log/standard_ppo_$DATE/standard_ppo_policy_sb3.zip" ]; then
    echo "❌ Error: Standard PPO policy not found!" | tee -a log_comparison_$DATE.txt
    echo "   Expected: log/standard_ppo_$DATE/standard_ppo_policy_sb3.zip" | tee -a log_comparison_$DATE.txt
    exit 1
fi

echo "✅ Both trained policies found. Starting evaluation..." | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

# Evaluate both policies on all benchmarks
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== Evaluating on $benchmark_name ($benchmark) ===" | tee -a log_comparison_$DATE.txt
    
    # Max-Entropy PPO Evaluation (now tested first - uses default behavior)
    echo "Testing Max-Entropy PPO on $benchmark_name..." | tee -a log_comparison_$DATE.txt
    python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --base-model-path "$BASE_MODEL_PATH" \
        --ea-model-path "$MODEL_PATH" \
        --model-id "eagle-max-entropy-ppo-$benchmark-$DATE" \
        --bench-name "$benchmark" \
        --use-online-rl \
        --use-sb3-discrete-ppo \
        --online-policy-path "log/max_entropy_ppo_$DATE/max_entropy_ppo_policy.pth" \
        --online-inference-only \
        --temperature 0.0 \
        --use_eagle3 \
        --max-new-token 512 \
        --num-choices 1 \
        --answer-file "log/max_entropy_ppo_$DATE/evaluation/${benchmark}_results.jsonl" \
        --no-wandb \
        2>&1 | tee log/max_entropy_ppo_$DATE/evaluation/${benchmark}_test.log
    
    # Standard PPO Evaluation (now tested second)
    echo "Testing Standard PPO on $benchmark_name..." | tee -a log_comparison_$DATE.txt
    python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
        --base-model-path "$BASE_MODEL_PATH" \
        --ea-model-path "$MODEL_PATH" \
        --model-id "eagle-standard-ppo-$benchmark-$DATE" \
        --bench-name "$benchmark" \
        --use-online-rl \
        --use-sb3-discrete-ppo \
        --disable-max-entropy \
        --online-policy-path "log/standard_ppo_$DATE/standard_ppo_policy.pth" \
        --online-inference-only \
        --temperature 0.0 \
        --use_eagle3 \
        --max-new-token 512 \
        --num-choices 1 \
        --answer-file "log/standard_ppo_$DATE/evaluation/${benchmark}_results.jsonl" \
        --no-wandb \
        2>&1 | tee log/standard_ppo_$DATE/evaluation/${benchmark}_test.log
    
    echo "✅ Completed evaluation on $benchmark_name" | tee -a log_comparison_$DATE.txt
    echo "" | tee -a log_comparison_$DATE.txt
done

echo "" | tee -a log_comparison_$DATE.txt
echo "=== Phase 4: Performance Analysis Across All Benchmarks ===" | tee -a log_comparison_$DATE.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

# Create consolidated results summary
echo "Performance Summary Report" > log_comparison_${DATE}_summary.txt
echo "=========================" >> log_comparison_${DATE}_summary.txt
echo "Training Date: $DATE" >> log_comparison_${DATE}_summary.txt
echo "Model: $MODEL_PATH" >> log_comparison_${DATE}_summary.txt
echo "Base Model: $BASE_MODEL_PATH" >> log_comparison_${DATE}_summary.txt
echo "" >> log_comparison_${DATE}_summary.txt

# Generate baseline results for all benchmarks (LLaMA 3.1 8B)
echo "=== Generating LLaMA 3.1 8B Baseline Results ===" | tee -a log_comparison_$DATE.txt
mkdir -p baseline_results

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Generating baseline for $benchmark..." | tee -a log_comparison_$DATE.txt
    
    # Generate EAGLE3 baseline
    if [ ! -f "baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" ]; then
        python -m eagle.evaluation.gen_ea_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" \
            --temperature 0.0 \
            --use_eagle3 \
            2>&1 | tee log/baseline_${benchmark}_eagle3.log
    else
        echo "EAGLE3 baseline for $benchmark already exists" | tee -a log_comparison_$DATE.txt
    fi
    
    # Generate standard baseline
    if [ ! -f "baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" ]; then
        python -m eagle.evaluation.gen_baseline_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" \
            --temperature 0.0 \
            2>&1 | tee log/baseline_${benchmark}_standard.log
    else
        echo "Standard baseline for $benchmark already exists" | tee -a log_comparison_$DATE.txt
    fi
done

# Analyze results for each benchmark with comprehensive comparisons
echo "" | tee -a log_comparison_$DATE.txt
echo "=== Comprehensive Performance Analysis ===" | tee -a log_comparison_$DATE.txt

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== $benchmark_name ($benchmark) Performance Analysis ===" | tee -a log_comparison_$DATE.txt
    echo "Benchmark: $benchmark_name ($benchmark)" >> log_comparison_${DATE}_summary.txt
    echo "===========================================" >> log_comparison_${DATE}_summary.txt
    
    # Define result files
    max_file="log/max_entropy_ppo_$DATE/evaluation/${benchmark}_results.jsonl"
    std_file="log/standard_ppo_$DATE/evaluation/${benchmark}_results.jsonl"
    eagle3_file="baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl"
    baseline_file="baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl"
    
    if [ -f "$max_file" ] && [ -f "$std_file" ] && [ -f "$eagle3_file" ] && [ -f "$baseline_file" ]; then
        echo "✅ All result files found for $benchmark_name" | tee -a log_comparison_$DATE.txt
        
        # Speed comparison using existing speed.py tool (3 comprehensive comparisons like test_expanded_action_space.sh)
        if [ -f "eagle/evaluation/speed.py" ]; then
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "1. Max-Entropy PPO vs EAGLE3 Baseline:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
            
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "2. Max-Entropy PPO vs Standard LLaMA Baseline:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
            
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "3. EAGLE3 Baseline vs Standard LLaMA Baseline:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$eagle3_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
            
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "4. Standard PPO vs EAGLE3 Baseline:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$eagle3_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
            
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "5. Standard PPO vs Standard LLaMA Baseline:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$std_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
            
            echo "" >> log_comparison_${DATE}_summary.txt
            echo "6. Max-Entropy PPO vs Standard PPO:" >> log_comparison_${DATE}_summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$max_file" \
                --baseline-file "$std_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log_comparison_${DATE}_summary.txt
                
        else
            echo "Speed analysis tool not found" >> log_comparison_${DATE}_summary.txt
        fi
        
        # Basic statistics
        echo "" >> log_comparison_${DATE}_summary.txt
        echo "Result File Statistics:" >> log_comparison_${DATE}_summary.txt
        echo "Max-Entropy PPO: $(wc -l < "$max_file") samples" >> log_comparison_${DATE}_summary.txt
        echo "Standard PPO: $(wc -l < "$std_file") samples" >> log_comparison_${DATE}_summary.txt
        echo "EAGLE3 Baseline: $(wc -l < "$eagle3_file") samples" >> log_comparison_${DATE}_summary.txt
        echo "Standard Baseline: $(wc -l < "$baseline_file") samples" >> log_comparison_${DATE}_summary.txt
        
    else
        echo "❌ Missing result files for $benchmark_name" | tee -a log_comparison_$DATE.txt
        echo "Required files:" | tee -a log_comparison_$DATE.txt
        echo "  Max-Entropy PPO: $max_file $([ -f "$max_file" ] && echo "✅" || echo "❌")" | tee -a log_comparison_$DATE.txt
        echo "  Standard PPO: $std_file $([ -f "$std_file" ] && echo "✅" || echo "❌")" | tee -a log_comparison_$DATE.txt
        echo "  EAGLE3 Baseline: $eagle3_file $([ -f "$eagle3_file" ] && echo "✅" || echo "❌")" | tee -a log_comparison_$DATE.txt
        echo "  Standard Baseline: $baseline_file $([ -f "$baseline_file" ] && echo "✅" || echo "❌")" | tee -a log_comparison_$DATE.txt
        echo "Missing result files - check evaluation and baseline generation logs" >> log_comparison_${DATE}_summary.txt
    fi
    
    echo "" >> log_comparison_${DATE}_summary.txt
    echo "" | tee -a log_comparison_$DATE.txt
done
echo "" | tee -a log_comparison_$DATE.txt
echo "=== Final Results Summary ===" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

echo "📁 Directory Structure:" | tee -a log_comparison_$DATE.txt
echo "├── log/max_entropy_ppo_$DATE/" | tee -a log_comparison_$DATE.txt
echo "│   ├── training.log                    # Training process log" | tee -a log_comparison_$DATE.txt
echo "│   ├── checkpoints/                    # Training checkpoints" | tee -a log_comparison_$DATE.txt
echo "│   ├── max_entropy_ppo_policy.pth      # Trained policy" | tee -a log_comparison_$DATE.txt
echo "│   └── evaluation/" | tee -a log_comparison_$DATE.txt
echo "│       ├── mt_bench_results.jsonl      # MT-Bench results" | tee -a log_comparison_$DATE.txt
echo "│       ├── humaneval_results.jsonl     # HumanEval results" | tee -a log_comparison_$DATE.txt
echo "│       ├── gsm8k_results.jsonl         # GSM8K results" | tee -a log_comparison_$DATE.txt
echo "│       ├── alpaca_results.jsonl        # Alpaca results" | tee -a log_comparison_$DATE.txt
echo "│       └── sum_results.jsonl           # CNN/DailyMail results" | tee -a log_comparison_$DATE.txt
echo "├── log/standard_ppo_$DATE/" | tee -a log_comparison_$DATE.txt
echo "│   ├── training.log                    # Training process log" | tee -a log_comparison_$DATE.txt
echo "│   ├── checkpoints/                    # Training checkpoints" | tee -a log_comparison_$DATE.txt
echo "│   ├── standard_ppo_policy.pth         # Trained policy" | tee -a log_comparison_$DATE.txt
echo "│   └── evaluation/" | tee -a log_comparison_$DATE.txt
echo "│       ├── mt_bench_results.jsonl      # MT-Bench results" | tee -a log_comparison_$DATE.txt
echo "│       ├── humaneval_results.jsonl     # HumanEval results" | tee -a log_comparison_$DATE.txt
echo "│       ├── gsm8k_results.jsonl         # GSM8K results" | tee -a log_comparison_$DATE.txt
echo "│       ├── alpaca_results.jsonl        # Alpaca results" | tee -a log_comparison_$DATE.txt
echo "│       └── sum_results.jsonl           # CNN/DailyMail results" | tee -a log_comparison_$DATE.txt
echo "├── baseline_results/" | tee -a log_comparison_$DATE.txt
echo "│   ├── mt_bench_LLaMA3.1-8B_eagle3.jsonl     # EAGLE3 baseline" | tee -a log_comparison_$DATE.txt
echo "│   ├── mt_bench_LLaMA3.1-8B_baseline.jsonl   # Standard baseline" | tee -a log_comparison_$DATE.txt
echo "│   ├── humaneval_LLaMA3.1-8B_eagle3.jsonl    # EAGLE3 baseline" | tee -a log_comparison_$DATE.txt
echo "│   ├── humaneval_LLaMA3.1-8B_baseline.jsonl  # Standard baseline" | tee -a log_comparison_$DATE.txt
echo "│   └── ... (all benchmarks × 2 baselines)" | tee -a log_comparison_$DATE.txt
echo "├── log_comparison_$DATE.txt             # This execution log" | tee -a log_comparison_$DATE.txt
echo "└── log_comparison_${DATE}_summary.txt   # Performance analysis summary" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

echo "🎯 Key Differences to Observe:" | tee -a log_comparison_$DATE.txt
echo "1. Parameter Diversity: Max-entropy should show more varied parameter combinations across all benchmarks" | tee -a log_comparison_$DATE.txt
echo "2. Exploration: Max-entropy should explore more of the action space during training" | tee -a log_comparison_$DATE.txt
echo "3. Consistency: Standard PPO should converge to more consistent parameter choices" | tee -a log_comparison_$DATE.txt
echo "4. Benchmark Performance: Compare speedup and accuracy across different task types" | tee -a log_comparison_$DATE.txt
echo "5. Wandb Logs: Compare entropy metrics and training curves between the two approaches" | tee -a log_comparison_$DATE.txt
echo "6. Comprehensive Comparisons: 6 different speed comparisons for thorough analysis" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

echo "📊 Comprehensive Speed Analysis (6 comparisons per benchmark):" | tee -a log_comparison_$DATE.txt
echo "1. Max-Entropy PPO vs EAGLE3 Baseline    (Our best method vs existing method)" | tee -a log_comparison_$DATE.txt
echo "2. Max-Entropy PPO vs Standard Baseline  (Our best method vs raw LLaMA)" | tee -a log_comparison_$DATE.txt
echo "3. EAGLE3 Baseline vs Standard Baseline  (Existing method vs raw LLaMA)" | tee -a log_comparison_$DATE.txt
echo "4. Standard PPO vs EAGLE3 Baseline       (Our standard method vs existing method)" | tee -a log_comparison_$DATE.txt
echo "5. Standard PPO vs Standard Baseline     (Our standard method vs raw LLaMA)" | tee -a log_comparison_$DATE.txt
echo "6. Max-Entropy PPO vs Standard PPO       (Our methods comparison)" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

echo "📊 Analysis Files:" | tee -a log_comparison_$DATE.txt
echo "- Main log: log_comparison_$DATE.txt" | tee -a log_comparison_$DATE.txt
echo "- Performance summary: log_comparison_${DATE}_summary.txt" | tee -a log_comparison_$DATE.txt
echo "- Wandb project: eagle-ppo-formal-comparison" | tee -a log_comparison_$DATE.txt
echo "" | tee -a log_comparison_$DATE.txt

echo "✅ Formal Training and Multi-Benchmark Evaluation Complete!" | tee -a log_comparison_$DATE.txt
echo "📈 Both PPO policies have been trained and evaluated on all 5 benchmarks" | tee -a log_comparison_$DATE.txt
echo "� Comprehensive speed analysis with 6 comparisons per benchmark" | tee -a log_comparison_$DATE.txt
echo "�🔍 Check the summary file for detailed performance comparisons" | tee -a log_comparison_$DATE.txt

echo ""
echo "===================================================================="
echo "🎉 EAGLE PPO Formal Training & Evaluation Complete!"
echo "===================================================================="
echo ""
echo "📁 Main Results:"
echo "   - Execution log:     log_comparison_$DATE.txt"
echo "   - Performance summary: log_comparison_${DATE}_summary.txt"
echo ""
echo "📂 Max-Entropy PPO Results: log/max_entropy_ppo_$DATE/"
echo "📂 Standard PPO Results:  log/standard_ppo_$DATE/"
echo "📂 Baseline Results: baseline_results/"
echo ""
echo "🔬 Benchmarks Evaluated:"
echo "   ✓ MT-Bench (conversation)"
echo "   ✓ HumanEval (coding)"
echo "   ✓ GSM8K (math)"
echo "   ✓ Alpaca (instructions)"
echo "   ✓ CNN/DailyMail (summarization)"
echo ""
echo "📊 Comprehensive Analysis:"
echo "   • 6 speed comparisons per benchmark"
echo "   • Our methods vs EAGLE3 baseline"
echo "   • Our methods vs standard LLaMA baseline"
echo "   • EAGLE3 vs standard baseline"
echo "   • Max-entropy vs Standard PPO"
echo ""
echo "📈 Analysis: Check log_comparison_${DATE}_summary.txt for detailed performance comparison"
echo "🌐 Wandb: eagle-ppo-formal-comparison project for training metrics"
echo "===================================================================="
