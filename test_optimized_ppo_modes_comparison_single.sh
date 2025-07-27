#!/bin/bash

# OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation
# This script demonstrates the optimized policy with:
# 1. EAGLE-3 layer features instead of SBERT text embeddings
# 2. Action caching to reduce computation frequency (every N steps)
# 3. NEW: Context-only state representation option (SBERT embeddings directly)
# 4. NEW: Choice between Standard and OFL policy versions
#
# USAGE:
# To run Standard version only: Set RUN_STANDARD_VERSION=1, RUN_OFL_VERSION=0
# To run OFL version only: Set RUN_STANDARD_VERSION=0, RUN_OFL_VERSION=1  
#
# Example configurations:
# - OFL only: RUN_STANDARD_VERSION=0, RUN_OFL_VERSION=1
#
#  --use-context-only-state \
DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_optimized_ppo"
# DATE='log/20250727_0412_optimized_ppo'
# DATE='20250725_0725_optimized_ppo'
# DATE='20250726_0904_optimized_ppo'
MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=4000

# EXECUTION MODE CONFIGURATION
# Set these to "true" or "false" to control which modes to run

# Policy implementation version selection
RUN_STANDARD_VERSION=1   # Run standard policy version
RUN_OFL_VERSION=1       # Run OFL policy version with enhanced features

# State representation modes
RUN_STANDARD=1          # Run without --use-context-only-state  
RUN_CONTEXT_ONLY=1      # Run with --use-context-only-state

# Entropy modes  
RUN_MAX_ENTROPY=1       # Run with max-entropy PPO
RUN_NO_MAX_ENTROPY=1    # Run without max-entropy (standard PPO)

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")
# BENCHMARKS=("gsm8k" "mt_bench")
# BENCHMARK_NAMES=("GSM8K" "MT-Bench")
# BENCHMARKS=("sum")
# BENCHMARK_NAMES=("CNN/DailyMail")
# # BENCHMARKS=("gsm8k")
# # BENCHMARKS=("gsm8k")

# Write execution config to each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    echo "=== 64 x 64 ===" | tee -a log/$DATE/$dir/execution_config.txt
    echo "=== EXECUTION MODE CONFIGURATION ===" | tee -a log/$DATE/$dir/execution_config.txt
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

# Create log directory - dynamic based on execution mode
DIRECTORIES_TO_CREATE=()

# Standard version directories
if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_context")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_context")
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_standard")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_standard")
        fi
    fi
fi

# OFL version directories
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_context_ofl")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_context_ofl")
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_standard_ofl")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_standard_ofl")
        fi
    fi
fi

# Create directories
if [ ${#DIRECTORIES_TO_CREATE[@]} -gt 0 ]; then
    # Create each directory individually to avoid brace expansion issues
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        mkdir -p log/$DATE/$dir
    done
    
    # Create subdirectories for each policy directory
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        mkdir -p log/$DATE/$dir/{checkpoints,evaluation,baseline_results}
    done
fi

# Function to write to a specific policy directory
write_to_policy_dir() {
    local policy_dir="$1"
    local message="$2"
    echo "$message" | tee -a log/$DATE/$policy_dir/comparison.txt
}

# Write comparison info to each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    write_to_policy_dir "$dir" "=== OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation ==="
    write_to_policy_dir "$dir" "Model: $MODEL_PATH"
    write_to_policy_dir "$dir" "Base Model: $BASE_MODEL_PATH"
    write_to_policy_dir "$dir" "Training Dataset: eagle/data/rl_training/question.jsonl"

    # Show which versions are being run
    if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
        write_to_policy_dir "$dir" "Policy Versions: BOTH (Standard + OFL with enhanced features)"
    elif [ "$RUN_OFL_VERSION" -eq 1 ]; then
        write_to_policy_dir "$dir" "Policy Version: OFL (with enhanced features)"
    elif [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
        write_to_policy_dir "$dir" "Policy Version: Standard"
    fi

    write_to_policy_dir "$dir" "OPTIMIZATION 1: EAGLE-3 layer features instead of SBERT embeddings"
    write_to_policy_dir "$dir" "OPTIMIZATION 2: Action caching - generate action every 10 steps"
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        write_to_policy_dir "$dir" "OPTIMIZATION 3: Context-only state representation (SBERT 384D directly)"
    fi
    write_to_policy_dir "$dir" "Expected speedup: ~50% reduction in RL policy computation"
    write_to_policy_dir "$dir" ""
done

# PHASE 1: STANDARD TRAINING (if enabled)
if [ "$RUN_STANDARD" -eq 1 ]; then
    # Standard Version Training
    if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
        # Phase 2a: Max-Entropy PPO (Standard) - if enabled
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            write_to_policy_dir "optimized_max_entropy_ppo_standard" ""
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "=== Phase 2a: Training with OPTIMIZED MAX-ENTROPY PPO (Standard) - Standard Version ==="
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- EAGLE-3 layer features for state representation"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- Action caching every 30 steps"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- High entropy coefficient 0.1 for exploration"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- Temperature-based inference T=1.5"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- Full feature state representation (EAGLE-3 + context)"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- Standard version"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" "- Training dataset: questions 0-$QUESTION_END for faster training"
            write_to_policy_dir "optimized_max_entropy_ppo_standard" ""

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_max_entropy_ppo_standard \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_max_entropy_ppo_standard/training_answers.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-sb3-discrete-ppo \
                --optimized-policy-version standard \
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
                --checkpoint-dir log/$DATE/optimized_max_entropy_ppo_standard/checkpoints \
                --online-policy-save-path log/$DATE/optimized_max_entropy_ppo_standard/optimized_max_entropy_ppo_policy_sb3.pt \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo_standard/training.log
        fi

        # Phase 2b: Standard PPO (Standard) - if enabled
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "" | tee -a log/$DATE/comparison.txt
            echo "=== Phase 2b: Training with OPTIMIZED STANDARD PPO (Standard) - Standard Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
            echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
            echo "- Full feature state representation (EAGLE-3 + context)" | tee -a log/$DATE/comparison.txt
            echo "- Standard version" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_standard_ppo_standard \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_standard_ppo_standard/training_answers.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-sb3-discrete-ppo \
                --optimized-policy-version standard \
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
                --disable-max-entropy \
                --action-cache-steps 10 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --checkpoint-dir log/$DATE/optimized_standard_ppo_standard/checkpoints \
                --online-policy-save-path log/$DATE/optimized_standard_ppo_standard/optimized_standard_ppo_policy_sb3.pt \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo_standard/training.log
        fi
    fi

    # OFL Version Training
    if [ "$RUN_OFL_VERSION" -eq 1 ]; then
        # Phase 2a: Max-Entropy PPO (Standard) - if enabled
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "" | tee -a log/$DATE/comparison.txt
            echo "=== Phase 2a: Training with OPTIMIZED MAX-ENTROPY PPO (Standard) - OFL Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- High entropy coefficient 0.1 for exploration" | tee -a log/$DATE/comparison.txt
            echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
            echo "- Full feature state representation (EAGLE-3 + context)" | tee -a log/$DATE/comparison.txt
            echo "- OFL version with enhanced features" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_max_entropy_ppo_standard_ofl \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_max_entropy_ppo_standard_ofl/training_answers.jsonl \
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
                --checkpoint-dir log/$DATE/optimized_max_entropy_ppo_standard_ofl/checkpoints \
                --online-policy-save-path log/$DATE/optimized_max_entropy_ppo_standard_ofl/optimized_max_entropy_ppo_policy_sb3.zip \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo_standard_ofl/training.log
        fi

        # Phase 2b: Standard PPO (Standard) - if enabled
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "" | tee -a log/$DATE/comparison.txt
            echo "=== Phase 2b: Training with OPTIMIZED STANDARD PPO (Standard) - OFL Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
            echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
            echo "- Full feature state representation (EAGLE-3 + context)" | tee -a log/$DATE/comparison.txt
            echo "- OFL version with enhanced features" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_standard_ppo_standard_ofl \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_standard_ppo_standard_ofl/training_answers.jsonl \
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
                --disable-max-entropy \
                --action-cache-steps 10 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --checkpoint-dir log/$DATE/optimized_standard_ppo_standard_ofl/checkpoints \
                --online-policy-save-path log/$DATE/optimized_standard_ppo_standard_ofl/optimized_standard_ppo_policy_sb3.zip \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo_standard_ofl/training.log
        fi
    fi
fi

# PHASE 2: CONTEXT-ONLY TRAINING (if enabled)
if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    # Standard Version Training
    if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
        # Phase 1a: Max-Entropy PPO (Context-Only) - if enabled
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "=== Phase 1a: Training with OPTIMIZED MAX-ENTROPY PPO (Context-Only) - Standard Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- High entropy coefficient 0.1 for exploration" | tee -a log/$DATE/comparison.txt
            echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
            echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
            echo "- Standard version" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_max_entropy_ppo_context \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_max_entropy_ppo_context/training_answers.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-sb3-discrete-ppo \
                --optimized-policy-version standard \
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
                --use-context-only-state \
                --checkpoint-dir log/$DATE/optimized_max_entropy_ppo_context/checkpoints \
                --online-policy-save-path log/$DATE/optimized_max_entropy_ppo_context/optimized_max_entropy_ppo_policy_sb3.pt \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo_context/training.log
        fi

        # Phase 1b: Standard PPO (Context-Only) - if enabled  
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "" | tee -a log/$DATE/comparison.txt
            echo "=== Phase 1b: Training with OPTIMIZED STANDARD PPO (Context-Only) - Standard Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
            echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
            echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
            echo "- Standard version" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_standard_ppo_context \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_standard_ppo_context/training_answers.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-sb3-discrete-ppo \
                --optimized-policy-version standard \
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
                --disable-max-entropy \
                --action-cache-steps 10 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --use-context-only-state \
                --checkpoint-dir log/$DATE/optimized_standard_ppo_context/checkpoints \
                --online-policy-save-path log/$DATE/optimized_standard_ppo_context/optimized_standard_ppo_policy_sb3.pt \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo_context/training.log
        fi
    fi

    # OFL Version Training
    if [ "$RUN_OFL_VERSION" -eq 1 ]; then
        # Phase 1a: Max-Entropy PPO (Context-Only) - if enabled
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "=== Phase 1a: Training with OPTIMIZED MAX-ENTROPY PPO (Context-Only) - OFL Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- High entropy coefficient 0.1 for exploration" | tee -a log/$DATE/comparison.txt
            echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
            echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
            echo "- OFL version with enhanced features" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_max_entropy_ppo_context_ofl \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_max_entropy_ppo_context_ofl/training_answers.jsonl \
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
                --use-context-only-state \
                --checkpoint-dir log/$DATE/optimized_max_entropy_ppo_context_ofl/checkpoints \
                --online-policy-save-path log/$DATE/optimized_max_entropy_ppo_context_ofl/optimized_max_entropy_ppo_policy_sb3.zip \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_ppo_context_ofl/training.log
        fi

        # Phase 1b: Standard PPO (Context-Only) - if enabled  
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "" | tee -a log/$DATE/comparison.txt
            echo "=== Phase 1b: Training with OPTIMIZED STANDARD PPO (Context-Only) - OFL Version ===" | tee -a log/$DATE/comparison.txt
            echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
            echo "- Action caching every 30 steps" | tee -a log/$DATE/comparison.txt
            echo "- Low entropy coefficient 0.01" | tee -a log/$DATE/comparison.txt
            echo "- Deterministic inference" | tee -a log/$DATE/comparison.txt
            echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
            echo "- OFL version with enhanced features" | tee -a log/$DATE/comparison.txt
            echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
            echo "" | tee -a log/$DATE/comparison.txt

            PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id optimized_standard_ppo_context_ofl \
                --question-file eagle/data/rl_training/question.jsonl \
                --question-begin 0 \
                --question-end $QUESTION_END \
                --answer-file log/$DATE/optimized_standard_ppo_context_ofl/training_answers.jsonl \
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
                --disable-max-entropy \
                --action-cache-steps 10 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --use-context-only-state \
                --checkpoint-dir log/$DATE/optimized_standard_ppo_context_ofl/checkpoints \
                --online-policy-save-path log/$DATE/optimized_standard_ppo_context_ofl/optimized_standard_ppo_policy_sb3.zip \
                --checkpoint-freq 500 \
                --wandb-project eagle-optimized-sb3-ppo \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_ppo_context_ofl/training.log
        fi
    fi
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt

# Count total number of models to evaluate
TOTAL_MODELS=0
if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # max-entropy context
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # standard context
    fi
fi
if [ "$RUN_STANDARD" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # max-entropy standard
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # standard standard
    fi
fi

echo "Testing $TOTAL_MODELS optimized trained policies on ${#BENCHMARKS[@]} benchmarks:" | tee -a log/$DATE/comparison.txt
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    echo "$((i+1)). $benchmark_name - $benchmark" | tee -a log/$DATE/comparison.txt
done
echo "" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist and create evaluation plan
POLICIES_TO_EVALUATE=()
POLICY_LABELS=()

# Standard Version Policies
if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_context/optimized_max_entropy_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_context")
            POLICY_LABELS+=("Max-Entropy PPO (Context-Only) - Standard")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only max-entropy policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_context/optimized_standard_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_context")
            POLICY_LABELS+=("Standard PPO (Context-Only) - Standard")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only standard policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_standard/optimized_max_entropy_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_standard")
            POLICY_LABELS+=("Max-Entropy PPO (Standard) - Standard")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard max-entropy policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_standard/optimized_standard_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_standard")
            POLICY_LABELS+=("Standard PPO (Standard) - Standard")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi
fi

# OFL Version Policies
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_context_ofl/optimized_max_entropy_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_context_ofl")
            POLICY_LABELS+=("Max-Entropy PPO (Context-Only) - OFL")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only max-entropy policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_context_ofl/optimized_standard_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_context_ofl")
            POLICY_LABELS+=("Standard PPO (Context-Only) - OFL")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only standard policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_standard_ofl/optimized_max_entropy_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_standard_ofl")
            POLICY_LABELS+=("Max-Entropy PPO (Standard) - OFL")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard max-entropy policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_standard_ofl/optimized_standard_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_standard_ofl")
            POLICY_LABELS+=("Standard PPO (Standard) - OFL")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi
fi

if [ ${#POLICIES_TO_EVALUATE[@]} -eq 0 ]; then
    echo "❌ No trained policies found for evaluation!" | tee -a log/$DATE/comparison.txt
    exit 1
fi

echo "✅ Found ${#POLICIES_TO_EVALUATE[@]} trained policies. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Evaluate all policies on all benchmarks
for j in "${!POLICIES_TO_EVALUATE[@]}"; do
    policy_dir="${POLICIES_TO_EVALUATE[$j]}"
    policy_label="${POLICY_LABELS[$j]}"
    
    for i in "${!BENCHMARKS[@]}"; do
        benchmark="${BENCHMARKS[$i]}"
        benchmark_name="${BENCHMARK_NAMES[$i]}"
        
        echo "=== Evaluating $benchmark_name with $policy_label ===" | tee -a log/$DATE/comparison.txt

        if [ ! -f log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl ]; then
            # Determine if this policy uses context-only mode
            CONTEXT_ARGS=""
            if [[ "$policy_dir" == *"_context" ]]; then
                CONTEXT_ARGS="--use-context-only-state"
            fi
            
            # Determine if this policy uses max-entropy mode
            ENTROPY_ARGS=""
            if [[ "$policy_dir" == *"max_entropy"* ]]; then
                ENTROPY_ARGS="--enable-max-entropy --inference-temperature 1.5 --max-entropy-inference"
            else
                ENTROPY_ARGS="--disable-max-entropy"
            fi
            
            # Determine policy version based on directory name
            POLICY_VERSION_ARG="standard"
            if [[ "$policy_dir" == *"_ofl" ]]; then
                POLICY_VERSION_ARG="ofl"
            fi
            
            python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id ${policy_dir}_$benchmark \
                --question-file eagle/data/$benchmark/question.jsonl \
                --question-begin 0 \
                --question-end -1 \
                --answer-file log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-sb3-discrete-ppo \
                --optimized-policy-version $POLICY_VERSION_ARG \
                --online-inference-only \
                $CONTEXT_ARGS \
                --online-policy-path log/$DATE/$policy_dir/optimized_*_ppo_policy_sb3.* \
                $ENTROPY_ARGS \
                --action-cache-steps 30 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/$policy_dir/evaluation/${benchmark}_evaluation.log
        else
            write_to_policy_dir "$policy_dir" "Results already exist for $policy_label on $benchmark_name"
        fi
    done
done


echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 4: Performance Analysis Across All Benchmarks ===" | tee -a log/$DATE/comparison.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create consolidated results summary for each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    echo "Performance Summary Report" >> log/$DATE/$dir/summary.txt
    echo "=========================" >> log/$DATE/$dir/summary.txt
    echo "Training Date: $DATE" >> log/$DATE/$dir/summary.txt
    echo "Model: $MODEL_PATH" >> log/$DATE/$dir/summary.txt
    echo "Base Model: $BASE_MODEL_PATH" >> log/$DATE/$dir/summary.txt
    echo "" >> log/$DATE/$dir/summary.txt
done

# Generate baseline results for all benchmarks (LLaMA 3.1 8B) - per policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    write_to_policy_dir "$dir" "=== Generating LLaMA 3.1 8B Baseline Results ==="

    for benchmark in "${BENCHMARKS[@]}"; do
        write_to_policy_dir "$dir" "Generating baseline for $benchmark..."
        
        # Generate EAGLE3 baseline
        if [ ! -f "log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" ]; then
            python -m eagle.evaluation.gen_ea_answer_llama3chat \
                --ea-model-path "$MODEL_PATH" \
                --base-model-path "$BASE_MODEL_PATH" \
                --bench-name "$benchmark" \
                --answer-file "log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" \
                --temperature 0.0 \
                --use_eagle3 \
                2>&1 | tee -a log/$DATE/$dir/baseline_results/baseline_${benchmark}_eagle3.log
        else
            write_to_policy_dir "$dir" "EAGLE3 baseline for $benchmark already exists"
        fi
    done
    
    for benchmark in "${BENCHMARKS[@]}"; do
        write_to_policy_dir "$dir" "Generating standard baseline for $benchmark..."
        
        # Generate standard baseline
        if [ ! -f "log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" ]; then
            python -m eagle.evaluation.gen_baseline_answer_llama3chat \
                --ea-model-path "$MODEL_PATH" \
                --base-model-path "$BASE_MODEL_PATH" \
                --bench-name "$benchmark" \
                --answer-file "log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" \
                --temperature 0.0 \
                2>&1 | tee -a log/$DATE/$dir/baseline_results/baseline_${benchmark}_standard.log
        else
            write_to_policy_dir "$dir" "Standard baseline for $benchmark already exists"
        fi
    done
done

# Analyze results for each benchmark with comprehensive comparisons - per policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    write_to_policy_dir "$dir" ""
    write_to_policy_dir "$dir" "=== Comprehensive Performance Analysis ==="

    for i in "${!BENCHMARKS[@]}"; do
        benchmark="${BENCHMARKS[$i]}"
        benchmark_name="${BENCHMARK_NAMES[$i]}"
        
        write_to_policy_dir "$dir" "=== $benchmark_name - $benchmark Performance Analysis ==="
        echo "Benchmark: $benchmark_name - $benchmark" >> log/$DATE/$dir/summary.txt
        echo "===========================================" >> log/$DATE/$dir/summary.txt
        
        # Define baseline files for this policy directory
        eagle3_file="log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl"
        baseline_file="log/$DATE/$dir/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl"
    
        # Create list of result files for this benchmark
        RESULT_FILES=()
        RESULT_LABELS=()
        
        # For each policy directory, check if it has results for this benchmark
        for policy_dir in "${DIRECTORIES_TO_CREATE[@]}"; do
            result_file="log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl"
            
            if [ -f "$result_file" ]; then
                RESULT_FILES+=("$result_file")
                RESULT_LABELS+=("$policy_dir")
            fi
        done
        
        if [ ${#RESULT_FILES[@]} -gt 0 ] && [ -f "$eagle3_file" ] && [ -f "$baseline_file" ]; then
            write_to_policy_dir "$dir" "✅ Found ${#RESULT_FILES[@]} policy results and baseline files for $benchmark_name"
        
        # Speed comparison using existing speed.py tool
        if [ -f "eagle/evaluation/speed.py" ]; then
            echo "" >> log/$DATE/$dir/summary.txt
            
            # Compare each policy against baselines
            for k in "${!RESULT_FILES[@]}"; do
                policy_file="${RESULT_FILES[$k]}"
                policy_label="${RESULT_LABELS[$k]}"
                
                echo "$((k*2+1)). $policy_label vs EAGLE3 Baseline:" >> log/$DATE/$dir/summary.txt
                python eagle/evaluation/speed.py \
                    --ea-file "$policy_file" \
                    --baseline-file "$eagle3_file" \
                    --tokenizer-path "$BASE_MODEL_PATH" \
                    2>&1 | tee -a log/$DATE/$dir/summary.txt
                
                echo "" >> log/$DATE/$dir/summary.txt
                echo "$((k*2+2)). $policy_label vs Standard LLaMA Baseline:" >> log/$DATE/$dir/summary.txt
                python eagle/evaluation/speed.py \
                    --ea-file "$policy_file" \
                    --baseline-file "$baseline_file" \
                    --tokenizer-path "$BASE_MODEL_PATH" \
                    2>&1 | tee -a log/$DATE/$dir/summary.txt
                echo "" >> log/$DATE/$dir/summary.txt
            done
            
            # Compare baselines against each other
            echo "EAGLE3 Baseline vs Standard LLaMA Baseline:" >> log/$DATE/$dir/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$eagle3_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/$dir/summary.txt
            
            # Compare policies against each other if we have multiple
            if [ ${#RESULT_FILES[@]} -gt 1 ]; then
                echo "" >> log/$DATE/$dir/summary.txt
                echo "Policy Comparisons:" >> log/$DATE/$dir/summary.txt
                for k in "${!RESULT_FILES[@]}"; do
                    for l in "${!RESULT_FILES[@]}"; do
                        if [ $k -lt $l ]; then
                            policy1_file="${RESULT_FILES[$k]}"
                            policy1_label="${RESULT_LABELS[$k]}"
                            policy2_file="${RESULT_FILES[$l]}"
                            policy2_label="${RESULT_LABELS[$l]}"
                            
                            echo "$policy1_label vs $policy2_label:" >> log/$DATE/$dir/summary.txt
                            python eagle/evaluation/speed.py \
                                --ea-file "$policy1_file" \
                                --baseline-file "$policy2_file" \
                                --tokenizer-path "$BASE_MODEL_PATH" \
                                2>&1 | tee -a log/$DATE/$dir/summary.txt
                            echo "" >> log/$DATE/$dir/summary.txt
                        fi
                    done
                done
            fi
                
        else
            echo "Speed analysis tool not found" >> log/$DATE/$dir/summary.txt
        fi
        
        # Basic statistics
        echo "" >> log/$DATE/$dir/summary.txt
        echo "Result File Statistics:" >> log/$DATE/$dir/summary.txt
        for k in "${!RESULT_FILES[@]}"; do
            policy_file="${RESULT_FILES[$k]}"
            policy_label="${RESULT_LABELS[$k]}"
            echo "$policy_label: $(wc -l < "$policy_file") samples" >> log/$DATE/$dir/summary.txt
        done
        echo "EAGLE3 Baseline: $(wc -l < "$eagle3_file") samples" >> log/$DATE/$dir/summary.txt
        echo "Standard Baseline: $(wc -l < "$baseline_file") samples" >> log/$DATE/$dir/summary.txt
        
    else
        write_to_policy_dir "$dir" "❌ Missing result files for $benchmark_name"
        write_to_policy_dir "$dir" "Required files:"
        for k in "${!RESULT_FILES[@]}"; do
            policy_file="${RESULT_FILES[$k]}"
            policy_label="${RESULT_LABELS[$k]}"
            write_to_policy_dir "$dir" "  $policy_label: $policy_file $([ -f "$policy_file" ] && echo "✅" || echo "❌")"
        done
        write_to_policy_dir "$dir" "  EAGLE3 Baseline: $eagle3_file $([ -f "$eagle3_file" ] && echo "✅" || echo "❌")"
        write_to_policy_dir "$dir" "  Standard Baseline: $baseline_file $([ -f "$baseline_file" ] && echo "✅" || echo "❌")"
        echo "Missing result files - check evaluation and baseline generation logs" >> log/$DATE/$dir/summary.txt
    fi
    
    echo "" >> log/$DATE/$dir/summary.txt
    write_to_policy_dir "$dir" ""
    done
done
done

# Count policies trained and evaluated for summary
TRAINED_POLICIES=0
if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi
if [ "$RUN_STANDARD" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi

# Determine execution mode description
EXECUTION_MODE=""
if [ "$RUN_CONTEXT_ONLY" -eq 1 ] && [ "$RUN_STANDARD" -eq 1 ]; then
    EXECUTION_MODE="Both States"
elif [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    EXECUTION_MODE="Context-Only"
elif [ "$RUN_STANDARD" -eq 1 ]; then
    EXECUTION_MODE="Standard"
fi

if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Both Entropy"
elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Max-Entropy"
elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Standard"
fi

# Add policy version information
if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Both Policy Versions"
elif [ "$RUN_OFL_VERSION" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + OFL Version"
elif [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Standard Version"
fi

# Write summary to each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    write_to_policy_dir "$dir" ""
    write_to_policy_dir "$dir" "=== Summary ==="
    write_to_policy_dir "$dir" "Training completed with optimized PPO policies!"
    write_to_policy_dir "$dir" "Execution Mode: $EXECUTION_MODE"
    write_to_policy_dir "$dir" "Policies trained: $TRAINED_POLICIES"
    write_to_policy_dir "$dir" "Results saved in: log/$DATE/$dir/"
    write_to_policy_dir "$dir" "Key optimizations implemented:"
    write_to_policy_dir "$dir" "1. EAGLE-3 layer features with optional context-only mode"
    write_to_policy_dir "$dir" "2. Action caching every 30 steps (~50% computation reduction)"
    write_to_policy_dir "$dir" "3. Flexible execution modes for targeted experiments"
    write_to_policy_dir "$dir" "4. Enhanced PPO with temperature-based action sampling"
    if [ "$RUN_OFL_VERSION" -eq 1 ]; then
        write_to_policy_dir "$dir" "5. OFL version with enhanced features (set_max_timesteps, set_training_mode, enhanced PPO updates)"
    fi
    write_to_policy_dir "$dir" ""
done

# Create performance summary for each policy directory
for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
    echo "Performance Summary Report" >> log/$DATE/$dir/summary.txt
    echo "=========================" >> log/$DATE/$dir/summary.txt
    echo "Date: $DATE" >> log/$DATE/$dir/summary.txt
    echo "Execution Mode: $EXECUTION_MODE" >> log/$DATE/$dir/summary.txt
    echo "Policies Trained: $TRAINED_POLICIES" >> log/$DATE/$dir/summary.txt
    echo "Model: $MODEL_PATH" >> log/$DATE/$dir/summary.txt
    echo "Algorithm: Optimized PPO" >> log/$DATE/$dir/summary.txt
    echo "Optimizations: EAGLE-3 features + Action caching + Flexible execution" >> log/$DATE/$dir/summary.txt
    echo "Training questions: $QUESTION_END" >> log/$DATE/$dir/summary.txt
    echo "Benchmarks evaluated: ${BENCHMARKS[*]}" >> log/$DATE/$dir/summary.txt
    echo "Policies evaluated: ${#DIRECTORIES_TO_CREATE[@]}" >> log/$DATE/$dir/summary.txt
    for policy_dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        echo "  - $policy_dir" >> log/$DATE/$dir/summary.txt
    done
    echo "" >> log/$DATE/$dir/summary.txt

    write_to_policy_dir "$dir" "Check log/$DATE/$dir/summary.txt for detailed results."
done