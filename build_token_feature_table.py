import json
import os
import torch
import numpy as np
from model import SparseAutoencoder
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# Find the top 30 most highly activated positions in a feature vector
def find_non_zero(vec, topnum=30, disp=True):
    nonzero_indices = (vec != 0).nonzero(as_tuple=True)[0]
    nonzero_values = vec[nonzero_indices]

    topn_rel_indices = torch.topk(nonzero_values, k=min(topnum, len(nonzero_values))).indices
    topn_indices = nonzero_indices[topn_rel_indices]
    topn_values = nonzero_values[topn_rel_indices]

    if disp:
        for idx, val in zip(topn_indices.tolist(), topn_values.tolist()):
            print(f"Position: {idx}, Value: {val:.6f}")

    return topn_indices.cpu().numpy().tolist(), topn_values.cpu().numpy().tolist()


# Extracts and saves average sparse feature vectors for each token using a trained Sparse Autoencoder (SAE).
def build_token_feature_table(sae_model, 
                              data_dir='./data/dataset.json',
                              output_dir='./data/token_feature_table.json'):
    
    if os.path.exists(output_dir):
        with open(output_dir, 'r', encoding='utf-8') as f:
            token_feature_table = json.load(f)
        return token_feature_table

    # Load the raw dataset containing stories, each with multiple tokens and associated vectors.
    with open(data_dir, 'r') as file:
        data_raw = json.load(file)

    # Dictionary to accumulate feature vectors per token_id
    token_feature_table = {}

    # Set the SAE model to evaluation mode (no dropout, no gradient updates)
    sae_model.eval()

    # Iterate over all stories in the dataset
    for k1, v1 in data_raw.items():
        for k2, v2 in v1.items():
            # Each v2[i] is a (token_id, activation) pair
            for i in range(len(v2)):
                with torch.no_grad():
                    # Encode the input vector using the SAE to get latent sparse vector z
                    _, z = sae_model(torch.Tensor(v2[i][1]).cuda())

                # Identify the non-zero (or top-n) indices and values in the sparse latent vector
                topn_indices, topn_values = find_non_zero(z, disp=False)

                # Get the token ID
                token_id = str(v2[i][0])

                # Initialize feature vector for this token ID if not already present
                if token_id not in token_feature_table:
                    token_feature_table[token_id] = np.zeros(sae_model.hidden_dim, dtype=np.float32)

                # Normalize the sparse activation values (like softmax)
                topn_values = np.array(topn_values, dtype=np.float32)
                topn_values = topn_values / (topn_values.sum() + 1e-8)

                # Accumulate the normalized activation values at the corresponding indices
                token_feature_table[token_id][topn_indices] += topn_values

    # Normalize the accumulated feature vector for each token so the sum equals 1
    for token_id, feature_vec in token_feature_table.items():
        token_feature_table[token_id] = (feature_vec / (feature_vec.sum() + 1e-8)).tolist()

    # Save the resulting token feature vectors to JSON file
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(token_feature_table, f, indent=4, ensure_ascii=False)

    print(f"Token feature extraction complete. Saved to {output_dir}")

    return token_feature_table


def plot_token_feature_distribution(token_feature_table, token_id):

    if token_id not in token_feature_table:
        print(f"Token ID {token_id} doesn't exist in dataset.")
        return

    feature_distribution = token_feature_table[token_id]
    feature_distribution = np.array(feature_distribution)

    smoothed = feature_distribution.copy()
    max_idx = np.argmax(feature_distribution)
    max_val = feature_distribution[max_idx]
    start = max(0, max_idx - 15)
    end = min(len(feature_distribution), max_idx + 16)
    smoothed[start:end] = max_val

    # import pdb; pdb.set_trace()

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(smoothed)), smoothed, color="red")
    plt.title(f"Feature Magnitude for Token ID {token_id} (Train)", fontsize=16)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, np.max(smoothed) * 1.1)
    plt.tight_layout()
    plt.savefig("./asset/activated_features_train.png", dpi=900)
    plt.close()


if __name__ == '__main__':

    sae_model = SparseAutoencoder(input_dim=896, hidden_dim=896*20).cuda()
    sae_model.load_state_dict(torch.load('./model/20250403-041718/best_model.pth'))
    token_feature_table = build_token_feature_table(sae_model)

    # print(f"{'car'}: 1803")
    # plot_token_feature_distribution(token_feature_table, '1803')

    print(f"{'train'}: 5426")
    plot_token_feature_distribution(token_feature_table, '5426')


