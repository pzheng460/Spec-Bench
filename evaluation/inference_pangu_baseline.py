"""Generate answers with OpenPanGu base model (autoregressive baseline).

Uses KVPanguForCausalLM with our custom KV cache for autoregressive
generation. Does NOT use HuggingFace generate() because our KV cache
is designed for in-place mutation (speculative decoding style).

Usage:
python3 evaluation/inference_pangu_baseline.py \
    --model-path /mnt/data/weights/openPangu-R-72B-2512/ \
    --model-id openpangu-72b-baseline \
    --bench-name spec_bench
"""
import argparse

import torch
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_eval, reorg_answer_file
from evaluation.device_utils import get_device, get_device_count, get_npu_device_map

from model.pangu.modeling_pangu_kv import KVPanguForCausalLM
from model.pangu.configs import PanguMTPConfig
from model.pangu.kv_cache import initialize_past_key_values
from transformers import AutoConfig, AutoTokenizer


def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, verbose=False):
    """Autoregressive generation using our custom KV cache (in-place mutation)."""
    import sys
    import time
    input_ids = inputs.input_ids

    # Initialize custom KV cache (4096 to support multi-turn prompts + generation)
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model, max_length=4096)
    current_length_data.zero_()

    with torch.inference_mode():
        # Prefill
        t0 = time.time()
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        if verbose:
            prefill_time = time.time() - t0
            print(f"\n  [Prefill] {input_ids.shape[1]} tokens in {prefill_time:.2f}s")

        # Decode token-by-token
        generated = input_ids
        decode_start = time.time()
        for step_i in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(outputs.logits[:, -1:] / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

            generated = torch.cat([generated, next_token], dim=-1)

            if verbose:
                token_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
                elapsed = time.time() - decode_start
                speed = (step_i + 1) / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\r  [Decode] {step_i+1} tokens | {speed:.1f} tok/s | {token_str!r}    ")
                sys.stdout.flush()

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)

        if verbose:
            total_decode = time.time() - decode_start
            n = generated.shape[1] - input_ids.shape[1]
            print(f"\n  [Done] {n} tokens in {total_decode:.2f}s ({n/total_decode:.1f} tok/s)")

    new_token = generated.shape[1] - input_ids.shape[1]
    step = new_token
    accept_length_list = [1] * new_token
    return generated, new_token, step, accept_length_list


def load_pangu_base_model(model_path, dtype=torch.float16, device_map="auto", max_memory=None):
    """Load OpenPanGu base model (layers 0-49) for autoregressive generation.

    Uses KVPanguForCausalLM.from_pretrained with accelerate meta-device
    init to avoid CPU OOM when creating the 72B model.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    base_config = PanguMTPConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        qk_nope_dim=getattr(config, 'qk_nope_dim', 128),
        qk_rope_dim=getattr(config, 'qk_rope_dim', 64),
        v_channels=getattr(config, 'v_channels', 128),
        param_sink_number=getattr(config, 'param_sink_number', 128),
        param_sink_with_value=getattr(config, 'param_sink_with_value', True),
        n_routed_experts=getattr(config, 'n_routed_experts', 80),
        n_shared_experts=getattr(config, 'n_shared_experts', 2),
        moe_intermediate_size=getattr(config, 'moe_intermediate_size', 1280),
        num_experts_per_tok=getattr(config, 'num_experts_per_tok', 8),
        first_k_dense_replace=getattr(config, 'first_k_dense_replace', 4),
        routed_scaling_factor=getattr(config, 'routed_scaling_factor', 2.5),
        norm_topk_prob=getattr(config, 'norm_topk_prob', True),
        router_enable_expert_bias=getattr(config, 'router_enable_expert_bias', True),
        sandwich_norm=getattr(config, 'sandwich_norm', True),
    )

    print("Loading base model with accelerate (meta-device init)...")
    model = KVPanguForCausalLM.from_pretrained(
        model_path,
        config=base_config,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/data/weights/openPangu-R-72B-2512/",
        help="Path to OpenPanGu model weights.",
    )
    parser.add_argument("--model-id", type=str, default="openpangu-72b-baseline")
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
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
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
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. OpenPanGu uses bfloat16.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show token-by-token generation progress during inference.",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    device = get_device()
    device_count = get_device_count(device)
    print(f"Detected device: {device}, count: {device_count}")

    max_memory = get_npu_device_map("60GiB") if device == "npu" else None
    if max_memory:
        print(f"NPU max_memory: {max_memory}")

    model = load_pangu_base_model(
        args.model_path,
        dtype=str_to_torch_dtype(args.dtype),
        device_map="auto",
        max_memory=max_memory,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build OpenPanGu prompt using the model's chat template
    def pangu_prompt_func(question_turns, turn_idx, previous_outputs):
        """Build prompt for OpenPanGu using its chat template."""
        messages = []
        for t in range(turn_idx + 1):
            messages.append({"role": "user", "content": question_turns[t]})
            if t < turn_idx:
                messages.append({"role": "assistant", "content": previous_outputs[t]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    # The model's EOS token is [unused10] (id 45892)
    pangu_stop_token_ids = [tokenizer.eos_token_id]
    # Stop string for post-processing
    pangu_stop_str = tokenizer.eos_token

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        verbose=args.verbose,
        prompt_func=pangu_prompt_func,
        stop_token_ids=pangu_stop_token_ids,
        stop_str=pangu_stop_str,
    )

    reorg_answer_file(answer_file)
