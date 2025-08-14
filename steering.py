import os
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F

import numpy as np
from transformers import pipeline, set_seed


def load_dataset(tokenizer):
    # with open("./gp_same_len.csv", "r") as f:
    with open("./nps_npz_mvrr_final_v3.csv", "r") as f:
        raw_data = [l.strip().split(",") for i, l in enumerate(f.readlines()) if i > 0]

        data = []
        amb = []
        gp = []
        non_gp = []

        for x in raw_data:
            data.append(
                dict(
                    ind=x[0],
                    sentence_type=x[1],
                    sentence_ambiguous=x[2],
                    sentence_gp=x[3],
                    sentence_non_gp=x[4],
                )
            )
            amb.append(x[2])
            gp.append(x[3])
            non_gp.append(x[4])

        input_amb = tokenizer(amb, padding=True, truncation=True, return_tensors="pt")
        input_gp = tokenizer(gp, padding=True, truncation=True, return_tensors="pt")
        input_non_gp = tokenizer(non_gp, padding=True, truncation=True, return_tensors="pt")

        print(len(data))
        print(data[0])

        diffs = []
        for i in range(len(data)):
            amb_tokens = input_amb['input_ids'][i].tolist()
            gp_tokens = input_gp['input_ids'][i].tolist()
            non_gp_tokens = input_non_gp['input_ids'][i].tolist()

            diff_index = -1
            for j in range(min(len(amb_tokens), len(gp_tokens), len(non_gp_tokens))):
                # mark the first position where GP and non-GP differ (vs ambiguous)
                if (amb_tokens[j] != gp_tokens[j]) or (amb_tokens[j] != non_gp_tokens[j]) or (
                        gp_tokens[j] != non_gp_tokens[j]):
                    diff_index = j
                    break
            data[i]["difference_index"] = diff_index
            diffs.append(diff_index)

        print(Counter(diffs))
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gpt2_xl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1.0, help="scale for steering vector addition")
    return parser.parse_args()


def get_layers(model):
    # Returns an ordered list of block modules and the count
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2 style
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):  # Gemma-2 style
        layers = model.model.layers
    else:
        raise ValueError("Unsupported model architecture for layer access.")
    return layers, len(layers)


def get_hidden_last_token_means(model, tokenizer, texts, device, batch_size=16):
    """Collect per-layer mean of last-token hidden states over `texts`."""
    all_sums = None
    n = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple: (layer0,...,layerN); each [B,S,D]
            # take last position per sequence (next-token context)
            last_hidden = [h[:, -1, :] for h in hs]  # list length = num_layers+1 (includes embeddings)
            # accumulate sums
            if all_sums is None:
                all_sums = [lh.sum(dim=0) for lh in last_hidden]
            else:
                for k in range(len(all_sums)):
                    all_sums[k] = all_sums[k] + last_hidden[k].sum(dim=0)
            n += last_hidden[0].shape[0]
    # convert to means
    means = [s / max(n, 1) for s in all_sums]
    return means  # includes embedding layer at index 0


def hook_add_last_token_vec(vec, alpha=1.0):
    """Return a forward hook that adds `alpha*vec` to the last-token residual."""

    def _hook(module, inputs, output):
        # output can be tensor or tuple(tensor, ...)
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
            x = x.clone()
            x[:, -1, :] = x[:, -1, :] + alpha * vec.to(x.device)
            return (x, *rest)
        else:
            x = output.clone()
            x[:, -1, :] = x[:, -1, :] + alpha * vec.to(x.device)
            return x

    return _hook


def mean_prob_gap(model, tokenizer, texts, gp_id, ngp_id, device, batch_size=16):
    """Mean next-token probability gap: P(gp) - P(non_gp) over `texts`."""
    vals = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)
            logits = out.logits[:, -1, :]  # [B, vocab]
            probs = F.softmax(logits, dim=-1)  # convert to probabilities
            vals.append((probs[:, gp_id] - probs[:, ngp_id]).detach().float().cpu())
    if len(vals) == 0:
        return float("nan")
    return torch.cat(vals, dim=0).mean().item()


def next_token_id(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"String '{s}' produced no tokens.")
    # we expect a single-token target for next-token probability
    if len(ids) > 1:
        print(f"[warn] '{s}' split into {len(ids)} tokens: {ids}. Using the first token id {ids[0]}.")
    return ids[0]


def strict_next_token_id(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, f"Expected single token for {s!r}, got {ids}"
    return ids[0]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == 'gpt2_xl':
        model_name = 'gpt2-xl'
    elif args.model_type == 'gemma2_2b':
        model_name = "google/gemma-2-2b"
    elif args.model_type == 'gemma2_2b_it':
        model_name = "google/gemma-2-2b-it"
    else:
        raise ValueError(f'Unrecognized model type: {args.model_type}')

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device,
    )
    model = pipe.model
    tokenizer = pipe.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    dataset = load_dataset(tokenizer)
    sentence_types = ["NPZ", "NPS"]

    # completion_tokens = {
    #     "gp": tokenizer(",")["input_ids"][0],  # GP marker
    #     "non_gp": tokenizer(" was")["input_ids"][0],  # non-GP marker
    # }
    completion_tokens = {
        "gp": next_token_id(tokenizer, ","),  # comma
        "non_gp": next_token_id(tokenizer, " was")  # leading space is important
    }
    for k, tid in completion_tokens.items():
        try:
            print(f"[target] {k}: id={tid} str={tokenizer.decode([tid])!r}")
        except Exception:
            print(f"[target] {k}: id={tid}")

    layers, num_layers = get_layers(model)

    # Per-sentence-type steering vectors per layer (index aligned with hidden_states list)
    # Note: hidden_states includes embeddings at idx 0; model blocks correspond to 1..num_layers
    diffs_in_means = {}

    for sentence_type in sentence_types:
        # --- compute difference-in-means vectors per layer (GP - nonGP) ---
        gp_texts = [d["sentence_gp"] for d in dataset if d["sentence_type"] == sentence_type]
        non_gp_texts = [d["sentence_non_gp"] for d in dataset if d["sentence_type"] == sentence_type]

        if len(gp_texts) == 0 or len(non_gp_texts) == 0:
            print(f"[warn] No samples for sentence_type={sentence_type}, skipping.")
            continue

        gp_means = get_hidden_last_token_means(model, tokenizer, gp_texts, device)
        ngp_means = get_hidden_last_token_means(model, tokenizer, non_gp_texts, device)

        # difference vectors for all hidden layers (embedding + blocks)
        diffs = [g - n for g, n in zip(gp_means, ngp_means)]
        diffs_in_means[sentence_type] = diffs  # length num_layers+1
        print(f"[info] Computed steering vectors for '{sentence_type}' ({len(diffs)} layers incl. embeddings).")

    # ---------- evaluate & plot ----------
    delta_by_type = {}  # sentence_type -> [delta per layer]
    baseline_by_type = {}
    steered_by_type = {}
    for sentence_type in sentence_types:
        if sentence_type not in diffs_in_means:
            continue

        amb_texts = [d["sentence_ambiguous"] for d in dataset if d["sentence_type"] == sentence_type]
        if len(amb_texts) == 0:
            print(f"[warn] No ambiguous samples for {sentence_type}, skipping.")
            continue

        gp_id = completion_tokens["gp"]
        ngp_id = completion_tokens["non_gp"]

        # Baseline (no steering)
        baseline_gap = mean_prob_gap(model, tokenizer, amb_texts, gp_id, ngp_id, device)
        baseline_by_type[sentence_type] = baseline_gap
        print(f"\n[baseline] {sentence_type}: mean(P(GP) - P(non-GP)) = {baseline_gap:.4f}")

        # Per-layer deltas
        vecs = diffs_in_means[sentence_type]  # includes embeddings at idx 0
        deltas = []
        steered_gaps = []

        for L in range(1, num_layers + 1):  # align block L-1 -> hidden_states[L]
            steer_vec = vecs[L]
            block = layers[L - 1]
            hook = block.register_forward_hook(hook_add_last_token_vec(steer_vec, alpha=args.alpha))

            steered_gap = mean_prob_gap(model, tokenizer, amb_texts, gp_id, ngp_id, device)
            hook.remove()

            steered_gaps.append(steered_gap)
            deltas.append(steered_gap - baseline_gap)
            print(
                f"[eval] {sentence_type} | layer {L:>3d}: steered={steered_gap:.4f}  Δ={steered_gap - baseline_gap:+.4f}")

        delta_by_type[sentence_type] = deltas
        steered_by_type[sentence_type] = steered_gaps

    # ---------- plotting ----------
    plt.figure(figsize=(9, 5))
    xs = np.arange(1, num_layers + 1)

    for sentence_type, deltas in delta_by_type.items():
        plt.plot(xs, deltas, marker='o', linewidth=1.5, label=f"{sentence_type} (Δ steered - baseline)")

    plt.axhline(0.0, linestyle='--', linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Change in gp vs non-gp score diff")
    plt.title("Per-layer change in next-token logit gap after steering")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("steering_delta_per_layer.png", dpi=1000)

    # (Optional) also plot absolute before/after for a quick sanity check
    plt.figure(figsize=(9, 5))
    for sentence_type, vals in steered_by_type.items():
        plt.plot(xs, vals, marker='o', linewidth=1.5, label=f"{sentence_type} steered")
        plt.plot(xs, [baseline_by_type[sentence_type]] * len(xs), linestyle='--', label=f"{sentence_type} baseline")
    plt.xlabel("Layer")
    plt.ylabel("mean P(GP) - P(non-GP)")
    plt.title("Baseline vs steered score gap by layer")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("steering_before_after.png", dpi=1000)


if __name__ == '__main__':
    main()
