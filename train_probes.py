import os
import argparse
from collections import Counter

from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

import pandas as pd
import umap

from transformers import pipeline, set_seed


def load_dataset(dataset_name, tokenizer):
    if dataset_name == 'gp_same_len.csv':
        with open(f"./{dataset_name}", "r") as f:
            raw_data = [l.strip().split(",") for i, l in enumerate(f.readlines()) if i > 0]
    else:
        df = pd.read_csv(f"./{dataset_name}", sep = ';', header = 'infer', skip_blank_lines=True)
        df.columns = ['ind', 'condition', 'sentence_ambiguous', 'sentence_gp', 'sentence_non_gp']
        raw_data = df.values.tolist()

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


def prepare_clustering_data(dataset):
    """
    Prepare data for clustering by combining all sentence types
    """
    sentences = []
    labels = []

    for item in dataset:
        # Add ambiguous sentence
        sentences.append(item['sentence_ambiguous'])
        labels.append('ambiguous')

        # TODO
        # Add GP sentence
        # sentences.append(item['sentence_gp'])
        # labels.append('gp')

        # Add non-GP sentence
        sentences.append(item['sentence_non_gp'])
        labels.append('non_gp')

    return sentences, labels


def map_clusters_to_labels(cluster_labels, true_labels):
    """
    Map cluster numbers to actual labels based on majority voting
    """
    cluster_to_label = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        cluster_true_labels = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
        if cluster_true_labels:  # Check if list is not empty
            most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
            print(most_common_label)
            cluster_to_label[cluster] = most_common_label
        else:
            cluster_to_label[cluster] = 'unknown'  # fallback

    # Map cluster labels to actual labels
    mapped_labels = [cluster_to_label[cluster] for cluster in cluster_labels]
    return mapped_labels, cluster_to_label


def eval_kmeans(dataset, tokenizer, model, device, trained_clfs, save_dir="./"):
    sentences, labels = prepare_clustering_data(dataset)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for k, v in inputs.items():
        inputs[k] = v.cpu()

    num_layers = len(outputs.hidden_states)
    accuracies = []
    plot_paths = []
    # Create directory for saving plots
    os.makedirs(save_dir, exist_ok=True)

    for layer in range(num_layers):
        layer_representations = outputs.hidden_states[layer]
        # Extract representations at differing token positions
        X = []
        for i in range(len(dataset)):
            # Get the difference index for this sample
            diff_pos = dataset[i]["difference_index"]
            # Extract features for all three sentence types for this sample
            for j, _ in enumerate(['ambiguous', 'gp', 'non_gp']):
                X.append(layer_representations[i + j, diff_pos, :].cpu().numpy())

        X = np.array(X)
        # Evaluate given k-means on the data
        kmeans, cluster_mapping = trained_clfs[layer]
        cluster_labels = kmeans.predict(X)
        # Map clusters to labels using the training mapping
        mapped_labels = [cluster_mapping.get(cluster, 'unknown') for cluster in cluster_labels]

        # Compute accuracy
        accuracy = np.mean([1 if true == pred else 0 for true, pred in zip(labels, mapped_labels)])
        accuracies.append(accuracy)
        print(f"Test: layer = {layer}, accuracy = {accuracy:.4f}")

        # Apply UMAP and generate a plot
        print("Umap", flush=True)
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(6, 5))
        unique_labels = list(sorted(set(labels)))
        colors = cm.get_cmap('tab10')
        # Plot each class with its own color
        for idx, lbl in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            plt.scatter(X_umap[indices, 0], X_umap[indices, 1],
                        label=str(lbl), color=colors(idx), s=10, alpha=0.7)

        plt.title(f"UMAP projection - Layer {layer} - Accuracy {accuracy:.2f}")
        plt.legend(markerscale=2)
        plt.tight_layout()

        # Save plot to disk and to the list
        plot_path = os.path.join(save_dir, f"layer_{layer}_umap.png")
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)

    # Generate a grid plot with all UMAP plots
    num_cols = 4
    num_rows = (num_layers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for idx, plot_path in enumerate(plot_paths):
        img = plt.imread(plot_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Layer {idx}\nAccuracy {accuracies[idx]:.2f}")

    # Turn off any unused subplots
    for j in range(len(plot_paths), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    grid_plot_path = os.path.join(save_dir, "grid_plot.png")
    plt.savefig(grid_plot_path)
    plt.close()

    print(f"Grid plot saved to {grid_plot_path}")
    return accuracies


def fit_kmeans(dataset, tokenizer, model, device, seed, n_clusters=3):
    sentences, labels = prepare_clustering_data(dataset)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for k, v in inputs.items():
        inputs[k] = v.cpu()

    num_layers = len(outputs.hidden_states)
    accuracies = []

    clfs = []
    for layer in range(num_layers):
        layer_representations = outputs.hidden_states[layer]
        # Extract representations at differing token positions
        X = []
        for i in range(len(dataset)):
            # Get the difference index for this sample
            diff_pos = dataset[i]["difference_index"]
            # Extract features for all three sentence types for this sample
            # TODO
            for j, _ in enumerate(['ambiguous', 'non_gp']):
                X.append(layer_representations[i + j, diff_pos, :].cpu().numpy())

        X = np.array(X)

        # Fit k-means with 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Map clusters to actual labels
        mapped_labels, cluster_mapping = map_clusters_to_labels(cluster_labels, labels)
        clfs.append((kmeans, cluster_mapping))

        # Compute accuracy
        accuracy = np.mean([1 if true == pred else 0 for true, pred in zip(labels, mapped_labels)])
        accuracies.append(accuracy)

        print(f"Train: layer = {layer}, accuracy = {accuracy:.4f}")
        print(f"Cluster mapping: {cluster_mapping}")

    return clfs, accuracies


def eval_binary_clfs(cls1_samples, cls2_samples, diff_indices, tokenizer, model, device, trained_clfs, save_dir="./"):
    inputs = tokenizer(cls1_samples + cls2_samples, padding=True, truncation=True, return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for k, v in inputs.items():
        inputs[k] = v.cpu()
    os.makedirs(save_dir, exist_ok=True)

    num_layers = len(outputs.hidden_states)
    accuracies = []
    plot_paths = []
    for layer in range(num_layers):
        layer_non_amb = outputs.hidden_states[layer][: len(cls1_samples)]
        layer_non_amb_labels = [0] * (len(cls2_samples))

        layer_amb = outputs.hidden_states[layer][len(cls1_samples):]
        layer_amb_labels = [1] * len(cls1_samples)

        # Extract representations at differing token positions
        X_non_amb = []
        X_amb = []

        for i in range(len(diff_indices)):
            # Get the hidden state at the differing position for this sample
            diff_pos = diff_indices[i]
            X_non_amb.append(layer_non_amb[i, diff_pos, :].cpu().numpy())
            X_amb.append(layer_amb[i, diff_pos, :].cpu().numpy())

        # Combine features and labels
        X = np.vstack([X_non_amb, X_amb])
        y = np.array(layer_non_amb_labels + layer_amb_labels)
        clf = trained_clfs[layer]

        # Compute accuracy on the same data (you might want to use cross-validation instead)
        # y_pred = clf.predict(X)
        # print("y_pred: ", y_pred, flush=True)
        # accuracy = np.mean(y == y_pred)
        y_proba = clf.predict_proba(X)[:, 1]  # in [0, 1]
        print("y_proba: ", y_proba, flush=True)
        y_pred = (y_proba >= 0.5).astype(int)  # optional: threshold to compute accuracy
        print("y_pred: ", y_pred, flush=True)
        accuracy = np.mean(y == y_pred)
        accuracies.append(accuracy)
        print(f"Test: layer = {layer}, accuracy = {accuracy:.4f}")

        # Apply UMAP and generate a plot
        print("Umap", flush=True)
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(6, 5))
        unique_labels = list(sorted(set(y)))
        colors = cm.get_cmap('tab10')

        # Plot each class with its own color
        for idx, lbl in enumerate(unique_labels):
            indices = [i for i, l in enumerate(y) if l == lbl]
            plt.scatter(X_umap[indices, 0], X_umap[indices, 1],
                        label=str(lbl), color=colors(idx), s=10, alpha=0.7)

        plt.title(f"UMAP projection - Layer {layer} - Accuracy {accuracy:.2f}")
        plt.legend(markerscale=2)
        plt.tight_layout()

        # Save plot to disk and to the list
        plot_path = os.path.join(save_dir, f"layer_{layer}_umap.png")
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)

    # Generate a grid plot with all UMAP plots
    num_cols = 4
    num_rows = (num_layers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for idx, plot_path in enumerate(plot_paths):
        img = plt.imread(plot_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Layer {idx}\nAccuracy {accuracies[idx]:.2f}")

    # Turn off any unused subplots
    for j in range(len(plot_paths), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    grid_plot_path = os.path.join(save_dir, "grid_plot.png")
    plt.savefig(grid_plot_path)
    plt.close()

    print(f"Grid plot saved to {grid_plot_path}")
    return accuracies


def fit_binary_clfs(cls1_samples, cls2_samples, diff_indices, tokenizer, model, device, seed):
    # input_cls1 = tokenizer(cls1_samples, padding=True, truncation=True, return_tensors="pt")
    # input_cls2 = tokenizer(cls2_samples, padding=True, truncation=True, return_tensors="pt")
    inputs = tokenizer(cls1_samples + cls2_samples, padding=True, truncation=True, return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for k, v in inputs.items():
        inputs[k] = v.cpu()

    num_layers = len(outputs.hidden_states)
    accuracies = []
    clfs = []

    for layer in range(num_layers):
        layer_non_amb = outputs.hidden_states[layer][: len(cls1_samples)]
        layer_non_amb_labels = [0] * (len(cls2_samples))

        layer_amb = outputs.hidden_states[layer][len(cls1_samples):]
        layer_amb_labels = [1] * len(cls1_samples)

        # Extract representations at differing token positions
        X_non_amb = []
        X_amb = []

        for i in range(len(diff_indices)):
            # Get the hidden state at the differing position for this sample
            diff_pos = diff_indices[i]
            X_non_amb.append(layer_non_amb[i, diff_pos, :].cpu().numpy())
            X_amb.append(layer_amb[i, diff_pos, :].cpu().numpy())

        # Combine features and labels
        X = np.vstack([X_non_amb, X_amb])
        y = np.array(layer_non_amb_labels + layer_amb_labels)

        # Train the probe
        # clf = svm.SVC(kernel='linear', random_state=seed)
        clf = svm.SVC(kernel='linear', probability=True, random_state=seed)  # enables predict_proba
        clf.fit(X, y)
        clfs.append(clf)

        # Compute accuracy on the same data (you might want to use cross-validation instead)
        # y_pred = clf.predict(X)
        # accuracy = np.mean(y == y_pred)
        y_proba = clf.predict_proba(X)[:, 1]  # in [0, 1]
        y_pred = (y_proba >= 0.5).astype(int)  # optional: threshold to compute accuracy
        accuracy = np.mean(y == y_pred)
        accuracies.append(accuracy)
        print(f"Train: layer = {layer}, accuracy = {accuracy:.4f}")

    return clfs, accuracies


def run_binary_eval_exp(cls1, cls2, train_dataset, test_dataset, sentence_type, tokenizer, model, device, save_dir,
                        args):
    train_amb = [x[cls1] for x in train_dataset if x["sentence_type"] == sentence_type]
    train_non_gp = [x[cls2] for x in train_dataset if x["sentence_type"] == sentence_type]
    train_diff_indices = [x["difference_index"] for x in train_dataset if x["sentence_type"] == sentence_type]
    clfs, train_accuracies = fit_binary_clfs(train_amb, train_non_gp, train_diff_indices, tokenizer, model, device,
                                             args.seed)

    test_amb = [x[cls1] for x in test_dataset if x["sentence_type"] == sentence_type]
    test_non_gp = [x[cls2] for x in test_dataset if x["sentence_type"] == sentence_type]
    test_diff_indices = [x["difference_index"] for x in test_dataset if x["sentence_type"] == sentence_type]
    test_accuracies = eval_binary_clfs(test_amb, test_non_gp, test_diff_indices, tokenizer, model, device, clfs,
                                       save_dir=save_dir)
    return test_accuracies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gpt2_xl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default='gp_same_len.csv')
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

    dataset = load_dataset(dataset_name, tokenizer)
    
    # clfs, train_accuracies = fit_kmeans(train_dataset, tokenizer, model, device, args.seed, n_clusters=2)
    # test_accuracies = eval_kmeans(test_dataset, tokenizer, model, device, clfs, save_dir=save_dir)

    for sentence_type in ["NPS", "NPZ", "MVRR"]:
        save_dir = f"./umap_plots_{args.model_type}"
        
        # dataset splitting
        if dataset_name == 'gp_same_len.csv':
            k = 16
        else:
            if sentence_type == "MVRR":
                k = 6
            else: 
                k = 8

        ds_by_type = [x for x in dataset if x["sentence_type"] == sentence_type]
        train_dataset = ds_by_type[:k]
        test_dataset = ds_by_type[k:]

        amb_gp = run_binary_eval_exp("sentence_ambiguous", "sentence_gp", train_dataset, test_dataset, sentence_type,
                                     tokenizer, model, device,
                                     save_dir, args)
        amb_non_gp = run_binary_eval_exp("sentence_ambiguous", "sentence_non_gp", train_dataset, test_dataset,
                                         sentence_type, tokenizer, model, device,
                                         save_dir, args)
        gp_non_gp = run_binary_eval_exp("sentence_non_gp", "sentence_gp", train_dataset, test_dataset, sentence_type,
                                        tokenizer, model, device,
                                        save_dir, args)

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(amb_gp)), amb_gp, marker='o', linestyle='-', color="blue", label="Ambiguous vs. GP")
        plt.plot(range(len(amb_non_gp)), amb_non_gp, marker='o', linestyle='-', color="red", label="Ambiguous vs. Non-GP")
        plt.plot(range(len(gp_non_gp)), gp_non_gp, marker='o', linestyle='-', color="green", label="Non-GP vs. GP")
        plt.ylim([0, 1.2])
        plt.xlabel("Layer")
        plt.ylabel("Test Accuracy")
        plt.title(f"Test Accuracies across Layers - {sentence_type} - last token ({args.model_type})")
        plt.grid(True)
        plt.legend(markerscale=2)
        plt.tight_layout()
        acc_plot_path = f"{save_dir}/test_accuracy_plot_{sentence_type}_last_token.png"
        print(acc_plot_path)
        plt.savefig(acc_plot_path)
        plt.close()


if __name__ == '__main__':
    main()
