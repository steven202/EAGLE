"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from .rl_tree_policy import RLTreePolicy, calculate_real_reward
    from .online_rl_policy import OnlineTreePolicy, calculate_online_reward
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
    from eagle.evaluation.rl_tree_policy import RLTreePolicy, calculate_real_reward
    from eagle.evaluation.online_rl_policy import OnlineTreePolicy, calculate_online_reward



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    # Initialize RL policy if requested
    rl_policy = None
    online_policy = None
    rl_data_entries = []
    
    if args.use_online_rl:
        # Initialize online RL policy for real-time learning
        print("Initializing Online RL Policy for real-time learning...")
        online_policy = OnlineTreePolicy(
            learning_rate=args.online_lr,
            epsilon_start=args.online_epsilon_start,
            epsilon_end=args.online_epsilon_end,
            memory_size=args.online_memory_size,
            batch_size=args.online_batch_size
        )
        
        # Try to load existing policy if specified
        if args.online_policy_path and os.path.exists(args.online_policy_path):
            online_policy.load(args.online_policy_path)
            print(f"Loaded existing online policy from {args.online_policy_path}")
        else:
            print("Starting with fresh online policy")
            
    elif args.use_rl_policy:
        # Use traditional offline RL policy
        try:
            rl_policy = RLTreePolicy(args.rl_policy_path)
            print(f"Loaded offline RL policy from {args.rl_policy_path}")
        except Exception as e:
            print(f"Failed to load RL policy: {e}")
            print("Falling back to fixed parameters from args")
            rl_policy = None

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # Get parameters from policy
            full_context = " ".join([msg["content"] for msg in messages])
            
            if online_policy is not None:
                # Online RL: predict parameters (inference mode during warmup)
                predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                    full_context, training_mode=False  # No exploration during warmup
                )
                print(f"Online RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            elif rl_policy is not None:
                # Offline RL policy
                predicted_total_tokens, predicted_depth, predicted_top_k = rl_policy.predict_parameters(full_context)
                print(f"Offline RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            else:
                # Fallback to defaults
                predicted_total_tokens = args.total_token
                predicted_depth = args.depth
                predicted_top_k = args.top_k
                print(f"Using default params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            
            torch.cuda.synchronize()
            start_time = time.time()

            # Use no_grad for model inference to save memory and computation
            with torch.no_grad():
                output_ids, new_token, idx = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    total_tokens=predicted_total_tokens,
                    depth=predicted_depth,
                    tree_top_k=predicted_top_k,
                )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()



            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                # Get parameters from policy
                full_context = " ".join([msg["content"] for msg in messages])
                
                if online_policy is not None:
                    # Online RL: predict parameters with appropriate training mode
                    if args.online_inference_only:
                        # Inference-only mode: no training, no exploration
                        predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                            full_context, training_mode=False
                        )
                        print(f"Online RL inference params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                    else:
                        # Training mode: enable exploration and learning
                        predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                            full_context, training_mode=True
                        )
                        print(f"Online RL training params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                elif rl_policy is not None:
                    # Offline RL policy
                    predicted_total_tokens, predicted_depth, predicted_top_k = rl_policy.predict_parameters(full_context)
                    print(f"Offline RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                else:
                    # Fallback to defaults
                    predicted_total_tokens = args.total_token
                    predicted_depth = args.depth
                    predicted_top_k = args.top_k
                    print(f"Using default params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")

                torch.cuda.synchronize()
                start_time = time.time()

                # Use no_grad for model inference to save memory and computation
                with torch.no_grad():
                    output_ids, new_token, idx = model.eagenerate(
                        torch.as_tensor(input_ids).cuda(),
                        temperature=temperature,
                        log=True,
                        is_llama3=True,
                        total_tokens=predicted_total_tokens,
                        depth=predicted_depth,
                        tree_top_k=predicted_top_k,
                    )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                # Online RL: Update policy with reward from this generation (if training enabled)
                if online_policy is not None and not args.online_inference_only:
                    # Convert tensor values to Python scalars
                    new_token_scalar = int(new_token.cpu()) if hasattr(new_token, 'cpu') else int(new_token)
                    
                    # Calculate reward for online learning
                    online_reward = calculate_online_reward(
                        total_time, new_token_scalar, predicted_total_tokens, 
                        predicted_depth, predicted_top_k
                    )
                    
                    # Update policy with this experience
                    online_policy.update_policy(online_reward)
                    
                    # Print learning progress periodically
                    if online_policy.step_count % 20 == 0:
                        stats = online_policy.get_performance_stats()
                        print(f"Online RL Update: Reward={online_reward:.3f}, "
                              f"Avg Recent Reward={stats.get('avg_reward_recent', 0):.3f}, "
                              f"Tokens/sec={new_token_scalar/total_time:.1f}")

                # Collect RL training data if requested
                if args.collect_rl_data:
                    # Convert tensor values to Python scalars
                    new_token_scalar = int(new_token.cpu()) if hasattr(new_token, 'cpu') else int(new_token)
                    rl_reward = calculate_real_reward(
                        total_time, new_token_scalar, predicted_total_tokens, 
                        predicted_depth, predicted_top_k
                    )
                    rl_data_entry = {
                        "question_id": question["question_id"],
                        "turn": j,
                        "choice": i,
                        "context": " ".join([msg["content"] for msg in messages[:-1]]),  # Exclude assistant response
                        "total_tokens": predicted_total_tokens,
                        "depth": predicted_depth,
                        "top_k": predicted_top_k,
                        "generation_time": total_time,
                        "new_tokens": new_token_scalar,
                        "reward": rl_reward,
                        "tokens_per_second": new_token_scalar / total_time if total_time > 0 else 0
                    }
                    rl_data_entries.append(rl_data_entry)

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    # Save online RL policy if used and training was enabled
    if online_policy is not None:
        if args.online_inference_only:
            # Inference-only mode: just show statistics
            final_stats = online_policy.get_performance_stats()
            print(f"\n=== Online RL Inference Complete ===")
            print(f"Used pre-trained policy for parameter optimization")
            if final_stats.get('most_used_params'):
                print(f"Most used parameters: {final_stats.get('most_used_params', [])}")
        else:
            # Training mode: save updated policy
            save_path = args.online_policy_save_path or "online_tree_policy_trained.pth"
            online_policy.save(save_path)
            
            # Print final statistics
            final_stats = online_policy.get_performance_stats()
            print(f"\n=== Online RL Training Complete ===")
            print(f"Total episodes: {final_stats.get('total_episodes', 0)}")
            print(f"Final average reward: {final_stats.get('avg_reward_recent', 0):.4f}")
            print(f"Most used parameters: {final_stats.get('most_used_params', [])}")
            print(f"Policy saved to: {save_path}")

    # Save RL training data if collected
    if args.collect_rl_data and rl_data_entries:
        rl_data_file = os.path.expanduser(args.rl_data_file)
        rl_data_dir = os.path.dirname(rl_data_file)
        if rl_data_dir:  # Only create directory if there's a directory path
            os.makedirs(rl_data_dir, exist_ok=True)
        with open(rl_data_file, "a") as fout:
            for entry in rl_data_entries:
                fout.write(json.dumps(entry) + "\n")
        print(f"Saved {len(rl_data_entries)} RL training entries to {rl_data_file}")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle3/llama31chat/8B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama31chat/8B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total-token = The total number of drafted tokens in the tree + 1. Used as fallback when RL policy is disabled or fails.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="depth = The maximum number of draft length - 1. Used as fallback when RL policy is disabled or fails.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of drafted tokens in each layer. Used as fallback when RL policy is disabled or fails.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use_eagle3",
        action="store_true"
    )
    parser.add_argument(
        "--use-rl-policy",
        action="store_true",
        help="Use RL policy to predict tree parameters dynamically"
    )
    parser.add_argument(
        "--rl-policy-path",
        type=str,
        default="ppo_tree_policy_discrete.zip",
        help="Path to trained RL policy model"
    )
    parser.add_argument(
        "--collect-rl-data",
        action="store_true",
        help="Collect data for RL training (save generation metrics)"
    )
    parser.add_argument(
        "--rl-data-file",
        type=str,
        default="rl_training_data.jsonl",
        help="File to save RL training data"
    )
    
    # Online RL arguments
    parser.add_argument(
        "--use-online-rl",
        action="store_true",
        help="Use online RL for real-time parameter optimization"
    )
    parser.add_argument(
        "--online-policy-path",
        type=str,
        help="Path to load existing online RL policy (optional)"
    )
    parser.add_argument(
        "--online-policy-save-path",
        type=str,
        default="online_tree_policy_trained.pth",
        help="Path to save trained online RL policy"
    )
    parser.add_argument(
        "--online-lr",
        type=float,
        default=3e-4,
        help="Learning rate for online RL policy"
    )
    parser.add_argument(
        "--online-epsilon-start",
        type=float,
        default=0.9,
        help="Initial exploration rate for online RL"
    )
    parser.add_argument(
        "--online-epsilon-end",
        type=float,
        default=0.05,
        help="Final exploration rate for online RL"
    )
    parser.add_argument(
        "--online-memory-size",
        type=int,
        default=1000,
        help="Experience replay memory size for online RL"
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        default=32,
        help="Batch size for online RL training"
    )
    parser.add_argument(
        "--online-inference-only",
        action="store_true",
        help="Use online RL policy for inference only (no training/learning)"
    )

    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}={v}")

    # Print helpful information about RL vs fixed parameters
    if args.use_online_rl:
        print("\n🚀 Online RL Mode: Real-time learning and parameter optimization")
        print(f"   Learning rate: {args.online_lr}")
        print(f"   Exploration: {args.online_epsilon_start} → {args.online_epsilon_end}")
        print(f"   Memory size: {args.online_memory_size}")
        print(f"   Policy will be saved to: {args.online_policy_save_path}")
        if args.online_policy_path:
            print(f"   Loading existing policy from: {args.online_policy_path}")
    elif args.use_rl_policy:
        print("\n🤖 Offline RL Policy Mode: Tree parameters will be predicted dynamically by RL policy")
        print(f"   Fallback parameters: total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")
        print(f"   RL policy path: {args.rl_policy_path}")
    else:
        print(f"\n⚙️  Fixed Parameter Mode: total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")
        if args.collect_rl_data:
            print("   Data collection enabled for future RL training")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
