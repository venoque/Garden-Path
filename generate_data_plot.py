import argparse
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import numpy as np
from collections import defaultdict

from transformers import pipeline, set_seed


def load_dataset(tokenizer):
    with open("./gp_same_len.csv", "r") as f:
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

            for j in range(len(amb_tokens)):
                if amb_tokens[j] != gp_tokens[j] != non_gp_tokens[j]:
                    diff_index = -1
                    data[i]["difference_index"] = diff_index
                    diffs.append(diff_index)
                    break

        print(Counter(diffs))
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gpt2_xl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda"

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

    dataset = load_dataset(tokenizer)
    sentence_types = ["NPZ", "NPS", "MVRR"]

    completion_tokens = {
        "gp": tokenizer(",")["input_ids"][0],  # GP marker
        "non_gp": tokenizer(" was")["input_ids"][0],  # non-GP marker (e.g., disambiguating verb)
    }
    results = defaultdict(lambda: {"Ambiguous": [], "GP": [], "Non-GP": []})

    for sentence_type in sentence_types:
        categories = {
            "Ambiguous": [x["sentence_ambiguous"] for x in dataset if x["sentence_type"] == sentence_type],
            "GP": [x["sentence_gp"] for x in dataset if x["sentence_type"] == sentence_type],
            "Non-GP": [x["sentence_non_gp"] for x in dataset if x["sentence_type"] == sentence_type],
        }

        for label, sents in categories.items():
            inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[:, -1, :]  # Take final token's logits (can adjust to target position if needed)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            gp_probs = probs[:, completion_tokens["gp"]]
            non_gp_probs = probs[:, completion_tokens["non_gp"]]
            diffs = (gp_probs - non_gp_probs).cpu().numpy()
            results[sentence_type][label].extend(diffs)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    fig.suptitle(f"Garden Path Continuation Probabilities ({model_name})", fontsize=14)

    colors = {'Ambiguous': '#F08080', 'GP': '#F5F5B0', 'Non-GP': '#87CEFA'}
    labels = ['Ambiguous', 'GP', 'Non-GP']

    for i, sentence_type in enumerate(sentence_types):
        ax = axs[i]
        means = [np.mean(results[sentence_type][label]) for label in labels]
        stds = [np.std(results[sentence_type][label]) / np.sqrt(len(results[sentence_type][label])) for label in labels]

        ax.bar(labels, means, yerr=stds, capsize=5, color=[colors[l] for l in labels])
        ax.set_title(sentence_type.replace("NPZ", "NP/Z").replace("NPS", "NP/S").replace("MVRR", "MV/RR"))
        ax.axhline(0, color="black", linewidth=0.5)
        if i == 0:
            ax.set_ylabel("p(GP) - p(non-GP)")
        ax.set_ylim(-0.25, 0.25)

    # Add curly brackets on final subplot
    axs[-1].annotate('GP', xy=(3.1, 0.13), xytext=(3.3, 0.13),
                     arrowprops=dict(arrowstyle='-[,widthB=2.5,lengthB=0.5', lw=1.5))
    axs[-1].annotate('non-GP', xy=(3.1, -0.05), xytext=(3.3, -0.05),
                     arrowprops=dict(arrowstyle='-[,widthB=3.5,lengthB=0.5', lw=1.5))

    fig.legend(handles=[plt.matplotlib.patches.Patch(color=colors[k], label=k) for k in labels],
               loc='lower center', ncol=3, title="Input Sentence Type")

    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(bottom=0.18, top=0.88)
    plt.savefig(f"./{args.model_type}_data.png", bbox_inches='tight', dpi=1000)


if __name__ == '__main__':
    main()
