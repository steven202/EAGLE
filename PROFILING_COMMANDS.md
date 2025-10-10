# EAGLE RL Profiling Test Commands

## Quick Test (2 questions only)
```bash
conda activate eagle-rl && export PYTHONPATH=/home/guo/EAGLE_RL_latency:$PYTHONPATH && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl_profiling \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --model-id optimized_max_entropy_ppo_profiling_test \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end 2 \
    --answer-file log/today/profiling_test/test_answers.jsonl \
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
    --checkpoint-dir log/today/profiling_test/checkpoints \
    --online-policy-save-path log/today/profiling_test/test_policy.zip \
    --checkpoint-freq 1 \
    --wandb-project eagle-profiling-test \
    --no-wandb \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-stepwise-rl \
    --use-eagle3
```

## Original Command Modified for Profiling (10 questions)
```bash
conda activate eagle-rl && export PYTHONPATH=/home/guo/EAGLE_RL_latency:$PYTHONPATH && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl_profiling \
    --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --model-id optimized_max_entropy_ppo_standard_ofl_profiling \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end 10 \
    --answer-file log/today/profiling/training_answers.jsonl \
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
    --checkpoint-dir log/today/profiling/checkpoints \
    --online-policy-save-path log/today/profiling/optimized_max_entropy_ppo_policy_sb3.zip \
    --checkpoint-freq 1 \
    --wandb-project eagle-optimized-sb3-ppo-profiling \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-stepwise-rl \
    --use-eagle3
```

## What to Look For in the Profiling Report

The profiling script will output a detailed breakdown showing:

1. **RL Policy Overhead**:
   - `rl_policy_prediction`: Time spent predicting parameters
   - `rl_policy_update`: Time spent updating the policy
   - `hidden_states_extraction`: Time spent extracting features for RL

2. **EAGLE Core Functions**:
   - `initialize_tree`: Tree setup for speculative generation
   - `tree_decoding`: Draft token generation
   - `evaluate_posterior`: Verification by target model  
   - `update_inference_inputs`: State updates between steps

3. **Model Operations**:
   - `model_generation`: Total generation time
   - `eagle_generation_core`: Core EAGLE algorithm
   - `generation_step_N`: Individual generation steps

4. **System Overhead**:
   - `model_initialization`: Loading models
   - `tokenization`: Text processing
   - `post_processing`: Output formatting

The report will categorize components and provide optimization suggestions based on where the most time is spent.