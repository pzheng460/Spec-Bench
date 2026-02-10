"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from evaluation.device_utils import get_device, get_visible_devices, device_synchronize


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
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
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):
    # Extract custom prompt builder and stop tokens if provided
    prompt_func = kwargs.pop("prompt_func", None)
    stop_token_ids = kwargs.pop("stop_token_ids", None)
    stop_str = kwargs.pop("stop_str", None)

    model.eval()
    print('Check model training state:', model.training)

    # Get device and visible devices info
    device = get_device()
    visible_devices = get_visible_devices()
    print(f'Using device: {device}')
    print(f'Visible devices: {visible_devices}')

    question = questions[0]

    def _build_prompt_and_generate(question_turns, j, previous_outputs, **gen_kwargs):
        """Build prompt for turn j and generate."""
        if prompt_func is not None:
            prompt = prompt_func(question_turns, j, previous_outputs)
            # Avoid double BOS: if template already includes BOS token, skip auto-adding it
            bos = getattr(tokenizer, "bos_token", None)
            add_special = not (bos and prompt.startswith(bos))
            inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=add_special).to(device)
        else:
            conv = get_conversation_template("vicuna")
            for t in range(j + 1):
                conv.append_message(conv.roles[0], question_turns[t])
                if t < j:
                    conv.append_message(conv.roles[1], previous_outputs[t])
                else:
                    conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
        return inputs

    def _post_process_output(output_ids, input_len):
        """Post-process generated output tokens."""
        output_ids = output_ids[0][input_len:]

        # Stop at stop_token_ids
        effective_stop_ids = stop_token_ids
        if effective_stop_ids is None and prompt_func is None:
            conv = get_conversation_template("vicuna")
            effective_stop_ids = conv.stop_token_ids
        if effective_stop_ids:
            stop_idx = [
                i for i, tid in enumerate(output_ids) if tid in effective_stop_ids
            ]
            if len(stop_idx) > 0:
                output_ids = output_ids[: stop_idx[0]]

        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)

        # Stop at stop_str
        effective_stop_str = stop_str
        if effective_stop_str is None and prompt_func is None:
            effective_stop_str = "</s>"
        if effective_stop_str and output.find(effective_stop_str) > 0:
            output = output[: output.find(effective_stop_str)]

        # Remove special tokens
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        if prompt_func is None:
            conv = get_conversation_template("vicuna")
            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

        return output

    # warmup — single prefill + 1 decode step only
    print("Warmup...")
    torch.manual_seed(0)
    inputs = _build_prompt_and_generate(question["turns"], 0, [])
    warmup_kwargs = dict(kwargs)
    warmup_kwargs.pop("verbose", None)
    device_synchronize(device)
    _, _, _, _ = forward_func(inputs, model, tokenizer, 1, **warmup_kwargs)  # max_new_tokens=1
    device_synchronize(device)
    print("Warmup done")

    accept_lengths_tree = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                inputs = _build_prompt_and_generate(question["turns"], j, turns)
                input_ids = inputs.input_ids
                try:
                    device_synchronize(device)
                    start_time = time.time()
                    output_ids, new_token, step, accept_length_tree = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                    device_synchronize(device)
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    output = _post_process_output(output_ids, len(input_ids[0]))
                except RuntimeError as e:
                    print(f"ERROR question ID: {question['question_id']}, turn {j}: {e}")
                    output = "ERROR"
                    new_token = 0
                    step = 0
                    total_time = 0.0
                    accept_length_tree = []

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                cur_accept_lengths_tree.extend(accept_length_tree)
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


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

