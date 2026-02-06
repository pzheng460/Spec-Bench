"""Tree buffer utilities for MTP speculative decoding."""

import copy
import torch
from evaluation.device_utils import get_device

TOPK = 10


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Args:
        path (list): The original list that needs padding.
        length (int): The desired length of the padded list.
        pad_value (int): The value to use for padding.

    Returns:
        list: Padded list.
    """
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device=None):
    """
    Generate tree buffers for speculative decoding.

    This function creates the necessary data structures for tree-based
    candidate generation and verification.

    Args:
        tree_choices: List of tree choice paths
        device: Target device for tensors

    Returns:
        dict: Tree buffers including attention mask, indices, and position ids
    """
    if device is None:
        device = get_device()

    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Count choices at each depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Build attention mask
    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0

    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Build tree indices
    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0

    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices

    # Build position IDs
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Build retrieve indices
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)

    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
        dim=1
    )

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Build repeat nums for tree structure
    repeat_nums = []
    for i in range(len(depth_counts)):
        if i == 0:
            repeat_nums.append([depth_counts[0]])
        else:
            nums = []
            start = sum(depth_counts[:i])
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j == 0:
                    nums.append(1)
                    parent = cur_parent
                elif cur_parent != parent:
                    nums.append(1)
                    parent = cur_parent
                else:
                    nums[-1] += 1
            repeat_nums.append(nums)

    # Aggregate the generated buffers
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
        "repeat_nums": repeat_nums,
    }

    # Move tensors to device
    tree_buffers = {
        k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    tree_buffers["attn_mask"] = [tree_buffers["tree_attn_mask"]] * len(depth_counts)

    return tree_buffers
