"""Generate answers using OpenPanGu MTP speculative decoding.

Usage:
python3 evaluation/inference_pangu_mtp.py \
    --base-model-path /mnt/data/weights/openPangu-R-72B-2512/ \
    --mtp-model-path /mnt/data/weights/openPangu-R-72B-2512/ \
    --model-id openpangu-72b-mtp \
    --bench-name spec_bench
"""
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file
from evaluation.device_utils import get_device, get_device_count, get_npu_device_map

from model.pangu.mtp_pangu_model import MTPPanguModel
from model.pangu.kv_cache import initialize_past_key_values
from model.mtp.utils import (
    generate_tree_buffers,
    initialize_tree,
    reset_tree_mode,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    prepare_logits_processor,
)
from model.eagle.choices import mc_sim_7b_63


def pangu_mtp_forward(inputs, model, tokenizer, max_new_tokens, tree_choices=None, logits_processor=None, max_steps=512):
    """
    Forward function for OpenPanGu MTP speculative decoding.

    Args:
        inputs: Tokenized inputs
        model: MTPPanguModel instance
        tokenizer: Tokenizer
        max_new_tokens: Maximum new tokens to generate
        tree_choices: Tree structure for speculation
        logits_processor: Processor for logits
        max_steps: Maximum decoding steps

    Returns:
        Tuple of (input_ids, new_token_count, step_count, accept_length_list)
    """
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    model.mtp_layer.reset_kv()
    accept_length_list = []

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices, device=model.base_model.model.layers[-1].self_attn.qkv_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.base_model.lm_head.weight.device)
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_tree_mode(model)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
    )
    new_token = 0

    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["tree_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
            tree_candidates, tree_buffers["b_indices"]
        )
        input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits_processor,
            logits,
            tree_logits,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state,
            hidden_state_new,
            sample_p
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            for i, id in enumerate(input_ids[0, input_len:]):
                if id == tokenizer.eos_token_id:
                    eos_token_ids_index = i
            invalid_len = len(input_ids[0, input_len:]) - eos_token_ids_index - 1
            if invalid_len > 0:
                accept_length_list[-1] -= invalid_len
                new_token -= invalid_len
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > 1960:
            break
    return input_ids, new_token, idx + 1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mtp-model-path",
        type=str,
        default="/mnt/data/weights/openPangu-R-72B-2512/",
        help="The path to the MTP model weights (same as base for OpenPanGu).",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/mnt/data/weights/openPangu-R-72B-2512/",
        help="The path to the base OpenPanGu model weights.",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="openpangu-72b-mtp")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec_bench",
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
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="The maximum number of decoding steps.",
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
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # Detect device and configure loading
    device = get_device()
    device_count = get_device_count(device)
    print(f"Detected device: {device}, count: {device_count}")

    # Limit per-GPU memory for base model to leave room for MTP layer + KV cache
    if device == "npu":
        max_memory = get_npu_device_map("60GiB")
    else:
        import torch
        max_memory = {i: "70GiB" for i in range(device_count)}
    print(f"max_memory: {max_memory}")

    model = MTPPanguModel.from_pretrained(
        base_model_path=args.base_model_path,
        mtp_model_path=args.mtp_model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = model.get_tokenizer()

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    # Build OpenPanGu prompt using the model's chat template
    def pangu_prompt_func(question_turns, turn_idx, previous_outputs):
        messages = []
        for t in range(turn_idx + 1):
            messages.append({"role": "user", "content": question_turns[t]})
            if t < turn_idx:
                messages.append({"role": "assistant", "content": previous_outputs[t]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    pangu_stop_token_ids = [tokenizer.eos_token_id]
    pangu_stop_str = tokenizer.eos_token

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=pangu_mtp_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        tree_choices=args.tree_choices,
        logits_processor=logits_processor,
        max_steps=args.max_steps,
        prompt_func=pangu_prompt_func,
        stop_token_ids=pangu_stop_token_ids,
        stop_str=pangu_stop_str,
    )

    reorg_answer_file(answer_file)
