#!/bin/bash
#SBATCH --job-name=continue_llama31_8b_4000_8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=62G
#SBATCH --time=72:00:00
#SBATCH --array=1
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/continue_llama31_8b_4000_8000_%A_%a.log
#SBATCH --mail-user=cwang33@wm.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --reservation=cwang33
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate /sciclone/home/cwang33/.conda/envs/eagle-rl

# CONTINUED TRAINING SCRIPT: LLaMA3.1-8B from 4000 to 8000 questions
# This script continues training from existing policies trained on 4000 questions
# and extends training to 8000 questions

DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_continued_llama31_8b_4000_8000"

# MODEL CONFIGURATION - LLaMA3.1-8B
MODEL_NAME="LLaMA3.1-8B"
MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
GEN_SCRIPT="gen_ea_answer_llama3chat_rl"
BASELINE_SCRIPT="gen_baseline_answer_llama3chat"
EAGLE3_SCRIPT="gen_ea_answer_llama3chat"

# TRAINING RANGE CONFIGURATION
QUESTION_BEGIN=0  # Start from question 4000 (where previous training ended)
QUESTION_END=8000    # Train up to question 8000

# NETWORK ARCHITECTURE CONFIGURATION - Based on original training
STANDARD_NET_ARCH="128,128"
OFL_NET_ARCH="128,128"

# EXECUTION MODE CONFIGURATION - Only run OFL version since that's what exists
RUN_STANDARD_VERSION=0   # No standard version policies found
RUN_OFL_VERSION=1       # Run OFL policy version with enhanced features
RUN_STANDARD=1          # Run without --use-context-only-state  
RUN_CONTEXT_ONLY=0      # No context-only policies found
RUN_MAX_ENTROPY=1       # Run with max-entropy PPO (this is what exists)
RUN_NO_MAX_ENTROPY=0    # No standard PPO policies found

# SOURCE DIRECTORY - Based on existing policies
SOURCE_DATE="20250729_184423_ofl128_llama318b"
SOURCE_BASE_DIR="log/results_llama3.1-8B/$SOURCE_DATE"

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")

# Create log directory for continued training
DIRECTORIES_TO_CREATE=()

# Only OFL version directories since that's what exists
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("continued_max_entropy_ppo_standard_ofl")
        fi
    fi
fi

# Create directories
if [ ${#DIRECTORIES_TO_CREATE[@]} -gt 0 ]; then
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        mkdir -p log/$DATE/$dir
    done
    
    # Create subdirectories for each policy directory
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        mkdir -p log/$DATE/$dir/{checkpoints,evaluation,baseline_results}
    done
    
    # Write execution config to each policy directory
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        echo "=== CONTINUED TRAINING CONFIGURATION ===" | tee -a log/$DATE/$dir/execution_config.txt
        echo "Source Training Date: $SOURCE_DATE" | tee -a log/$DATE/$dir/execution_config.txt
        echo "Continued Training Date: $DATE" | tee -a log/$DATE/$dir/execution_config.txt
        echo "Training Range: Questions $QUESTION_BEGIN to $QUESTION_END" | tee -a log/$DATE/$dir/execution_config.txt
        echo "Model: $MODEL_NAME" | tee -a log/$DATE/$dir/execution_config.txt
        echo "POLICY VERSIONS:" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_STANDARD_VERSION: $RUN_STANDARD_VERSION" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_OFL_VERSION: $RUN_OFL_VERSION" | tee -a log/$DATE/$dir/execution_config.txt
        echo "STATE MODES:" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_CONTEXT_ONLY: $RUN_CONTEXT_ONLY" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_STANDARD: $RUN_STANDARD" | tee -a log/$DATE/$dir/execution_config.txt
        echo "ENTROPY MODES:" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_MAX_ENTROPY: $RUN_MAX_ENTROPY" | tee -a log/$DATE/$dir/execution_config.txt
        echo "RUN_NO_MAX_ENTROPY: $RUN_NO_MAX_ENTROPY" | tee -a log/$DATE/$dir/execution_config.txt
        echo "" | tee -a log/$DATE/$dir/execution_config.txt
    done
fi

# Create fresh summary.txt files for each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    echo "Continued Training Performance Summary Report" > log/$DATE/$dir/summary.txt
    echo "=============================================" >> log/$DATE/$dir/summary.txt
    echo "Continued Training Date: $DATE" >> log/$DATE/$dir/summary.txt
    echo "Source Training Date: $SOURCE_DATE" >> log/$DATE/$dir/summary.txt
    echo "Training Range: Questions $QUESTION_BEGIN to $QUESTION_END" >> log/$DATE/$dir/summary.txt
    echo "Model: $MODEL_PATH" >> log/$DATE/$dir/summary.txt
    echo "Base Model: $BASE_MODEL_PATH" >> log/$DATE/$dir/summary.txt
    echo "" >> log/$DATE/$dir/summary.txt
done

# copy the source policy checkpoint to the continued training directory
cp -rf $SOURCE_BASE_DIR/optimized_max_entropy_ppo_standard_ofl/checkpoints/* log/$DATE/continued_max_entropy_ppo_standard_ofl/checkpoints/

# PHASE 1: STANDARD TRAINING (if enabled)
if [ "$RUN_STANDARD" -eq 1 ]; then
    # OFL Version Training
    if [ "$RUN_OFL_VERSION" -eq 1 ]; then
        # Phase 2a: Max-Entropy PPO (Standard) - if enabled
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "" 
            echo "=== Phase 2a: Continued Training with OPTIMIZED MAX-ENTROPY PPO (Standard) - OFL Version ===" 
            echo "- Loading existing policy from $SOURCE_DATE" 
            echo "- Training on questions $QUESTION_BEGIN to $QUESTION_END" 
            echo "- EAGLE-3 layer features for state representation" 
            echo "- Action caching every 30 steps" 
            echo "- High entropy coefficient 0.1 for exploration" 
            echo "- Temperature-based inference T=1.5" 
            echo "- Full feature state representation (EAGLE-3 + context)" 
            echo "- OFL version with enhanced features" 
            echo "" 

            # Check if source policy exists
            SOURCE_POLICY="$SOURCE_BASE_DIR/optimized_max_entropy_ppo_standard_ofl/optimized_max_entropy_ppo_policy_sb3.zip"
            if [ ! -f "$SOURCE_POLICY" ]; then
                echo "❌ Source policy not found: $SOURCE_POLICY"
                echo "Please check the SOURCE_DATE and SOURCE_BASE_DIR variables"
                exit 1
            fi
            # echo "RUNNING command: "
            # echo "PYTHONUNBUFFERED=1 python -m eagle.evaluation.$GEN_SCRIPT \
            #     --ea-model-path $MODEL_PATH \
            #     --base-model-path $BASE_MODEL_PATH \
            #     --model-id continued_max_entropy_ppo_standard_ofl \
            #     --question-file eagle/data/rl_training/question.jsonl \
            #     --question-begin $QUESTION_BEGIN \
            #     --question-end $QUESTION_END \
            #     --answer-file log/$DATE/continued_max_entropy_ppo_standard_ofl/training_answers.jsonl \
            #     --num-choices 1 \
            #     --num-gpus-per-model 1 \
            #     --num-gpus-total 1 \
            #     --max-gpu-memory \"80GiB\" \
            #     --dtype float16 \
            #     --temperature 0.0 \
            #     --use-online-rl \
            #     --use-optimized-sb3-discrete-ppo \
            #     --optimized-policy-version ofl \
            #     --online-lr 3e-4 \
            #     --ppo-n-steps 64 \
            #     --ppo-batch-size 32 \
            #     --ppo-epochs 4 \
            #     --ppo-gamma 0.95 \
            #     --ppo-gae-lambda 0.9 \
            #     --ppo-clip-range 0.2 \
            #     --ppo-vf-coef 0.5 \
            #     --ppo-ent-coef 0.01 \
            #     --max-grad-norm 0.5 \
            #     --enable-max-entropy \
            #     --max-entropy-ent-coef 0.1 \
            #     --inference-temperature 1.5 \
            #     --max-entropy-inference \
            #     --action-cache-steps 10 \
            #     --action-cache-enabled \
            #     --use-eagle3-features \
            #     --hidden-size 4096 \
            #     --ppo-net-arch \"$OFL_NET_ARCH\" \
            #     --checkpoint-dir log/$DATE/continued_max_entropy_ppo_standard_ofl/checkpoints \
            #     --online-policy-save-path log/$DATE/continued_max_entropy_ppo_standard_ofl/continued_max_entropy_ppo_policy_sb3.zip \
            #     --checkpoint-freq 500 \
            #     --wandb-project eagle-optimized-sb3-ppo \
            #     --total-token 60 \
            #     --depth 7 \
            #     --top-k 10 \
            #     --use-stepwise-rl \
            #     --use-eagle3 \
            #     --online-policy-path \"$SOURCE_POLICY\" 2>&1 | tee -a log/$DATE/continued_max_entropy_ppo_standard_ofl/training.log"
            echo "✅ Found source policy: $SOURCE_POLICY"
            echo "Starting continued training..."

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.$GEN_SCRIPT \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id continued_max_entropy_ppo_standard_ofl \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin $QUESTION_BEGIN \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/continued_max_entropy_ppo_standard_ofl/training_answers.jsonl \
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
                --ppo-net-arch "$OFL_NET_ARCH" \
                --checkpoint-dir log/$DATE/continued_max_entropy_ppo_standard_ofl/checkpoints \
                --online-policy-save-path log/$DATE/continued_max_entropy_ppo_standard_ofl/continued_max_entropy_ppo_policy_sb3.zip \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 \
                --online-policy-path "$SOURCE_POLICY" 2>&1 | tee -a log/$DATE/continued_max_entropy_ppo_standard_ofl/training.log
        fi
    fi
fi

echo ""
echo "=== Phase 3: Evaluation of Continued Policy (LLaMA3.1-8B) ==="

# Evaluate the continued policy on all benchmarks and generate baselines
policy_dir="continued_max_entropy_ppo_standard_ofl"
mkdir -p "log/$DATE/$policy_dir/evaluation" "log/$DATE/$policy_dir/baseline_results"

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    echo "--- Evaluating $benchmark_name ($benchmark) ---"

    # 1) Our method (continued policy)
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.$GEN_SCRIPT \
        --ea-model-path "$MODEL_PATH" \
        --base-model-path "$BASE_MODEL_PATH" \
        --model-id ${policy_dir}_$benchmark \
        --question-file eagle/data/$benchmark/question.jsonl \
        --question-begin 0 \
        --answer-file log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory "80GiB" \
        --dtype float16 \
        --temperature 0.0 \
        --use-online-rl \
        --use-optimized-sb3-discrete-ppo \
        --optimized-policy-version ofl \
        --online-inference-only \
        --online-policy-path log/$DATE/$policy_dir/continued_max_entropy_ppo_policy_sb3.zip \
        --enable-max-entropy --inference-temperature 1.5 --max-entropy-inference \
        --action-cache-steps 30 \
        --action-cache-enabled \
        --use-eagle3-features \
        --hidden-size 4096 \
        --ppo-net-arch "$OFL_NET_ARCH" \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-stepwise-rl \
        --use-eagle3 2>&1 | tee -a log/$DATE/$policy_dir/evaluation/${benchmark}_evaluation.log

    # 2) EAGLE3 baseline
    python -m eagle.evaluation.$EAGLE3_SCRIPT \
        --ea-model-path "$MODEL_PATH" \
        --base-model-path "$BASE_MODEL_PATH" \
        --bench-name "$benchmark" \
        --answer-file "log/$DATE/$policy_dir/baseline_results/${benchmark}_${MODEL_NAME}_eagle3.jsonl" \
        --temperature 0.0 \
        --use_eagle3 2>&1 | tee -a log/$DATE/$policy_dir/baseline_results/baseline_${benchmark}_eagle3.log

    # 3) Standard baseline
    python -m eagle.evaluation.$BASELINE_SCRIPT \
        --ea-model-path "$MODEL_PATH" \
        --base-model-path "$BASE_MODEL_PATH" \
        --bench-name "$benchmark" \
        --answer-file "log/$DATE/$policy_dir/baseline_results/${benchmark}_${MODEL_NAME}_baseline.jsonl" \
        --temperature 0.0 2>&1 | tee -a log/$DATE/$policy_dir/baseline_results/baseline_${benchmark}_standard.log
done

echo ""
echo "=== Phase 4: Performance Analysis (LLaMA3.1-8B) ==="
summary_file="log/$DATE/$policy_dir/summary.txt"
echo "Benchmark Comparisons:" >> "$summary_file"
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    policy_file="log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl"
    eagle3_file="log/$DATE/$policy_dir/baseline_results/${benchmark}_${MODEL_NAME}_eagle3.jsonl"
    baseline_file="log/$DATE/$policy_dir/baseline_results/${benchmark}_${MODEL_NAME}_baseline.jsonl"
    if [ -f "eagle/evaluation/speed.py" ]; then
        echo "" >> "$summary_file"
        echo "$benchmark: Continued vs EAGLE3" >> "$summary_file"
        python eagle/evaluation/speed.py --ea-file "$policy_file" --baseline-file "$eagle3_file" --tokenizer-path "$BASE_MODEL_PATH" 2>&1 | tee -a "$summary_file"
        echo "" >> "$summary_file"
        echo "$benchmark: Continued vs Standard Baseline" >> "$summary_file"
        python eagle/evaluation/speed.py --ea-file "$policy_file" --baseline-file "$baseline_file" --tokenizer-path "$BASE_MODEL_PATH" 2>&1 | tee -a "$summary_file"
    fi
done

echo "" 
echo "=== Continued Training Summary ===" 
echo "Continued training completed from questions $QUESTION_BEGIN to $QUESTION_END!" 
echo "Source training date: $SOURCE_DATE" 
echo "Continued training date: $DATE" 
echo "Results saved in: log/$DATE/" 
echo "Key features:" 
echo "1. Loaded existing OFL max-entropy policy from $SOURCE_DATE" 
echo "2. Continued training on questions $QUESTION_BEGIN-$QUESTION_END" 
echo "3. Maintained all original training configurations" 
echo "4. New policy saved as continued_max_entropy_ppo_policy_sb3.zip" 
echo ""
