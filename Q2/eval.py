import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from dataset import LibriSpeechDataset
from models.disentangler import SpeakerNet

def compute_eer(labels, scores):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # The EER is the point where fpr == fnr
    interp_fnr = interp1d(fpr, fnr)
    eer = brentq(lambda x: x - interp_fnr(x), 0, 1)
    return eer

def evaluate():
    with open("configs/config.json", "r") as f: 
        config = json.load(f)
    
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Real Test Data
    # Note: For evaluation, we use the dataset logic to get real embeddings
    test_dataset = LibriSpeechDataset(config["data_path"], config["feature_dim"])
    # We only need a subset for verification pairs to keep it fast
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    modes = ["baseline", "disentangled", "improved"]
    results = {}

    for mode in modes:
        checkpoint_path = f"checkpoints/{mode}.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {mode}: Checkpoint not found.")
            continue

        model = SpeakerNet(config, mode).to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        all_embeddings = []
        all_labels = []

        print(f"Extracting embeddings for {mode}...")
        with torch.no_grad():
            # Process a limited number of batches for verification pairs (e.g., 2000 samples)
            for i, (batch_x, batch_y_spk, _) in enumerate(test_loader):
                if i > 60: break 
                batch_x = batch_x.to(device)
                _, _, spk_emb, _ = model(batch_x)
                all_embeddings.append(spk_emb.cpu())
                all_labels.append(batch_y_spk.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # 2. Generate Verification Pairs (Positive and Negative)
        scores = []
        y_true = []
        num_samples = len(labels)

        print(f"Computing Verification Scores for {mode}...")
        for i in range(min(num_samples, 1000)):
            # Positive pair (Same speaker)
            same_idx = np.where(labels == labels[i])[0]
            if len(same_idx) > 1:
                j = np.random.choice(same_idx)
                if i != j:
                    sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
                    scores.append(sim.item())
                    y_true.append(1)

            # Negative pair (Different speaker)
            diff_idx = np.where(labels != labels[i])[0]
            k = np.random.choice(diff_idx)
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[k].unsqueeze(0))
            scores.append(sim.item())
            y_true.append(0)

        # 3. Calculate EER
        eer_val = compute_eer(y_true, scores)
        results[mode] = eer_val * 100 # Convert to percentage
        print(f"Result for {mode}: EER = {results[mode]:.2f}%")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['#ff7f0e', '#1f77b4', '#2ca02c'])
    plt.ylabel("Equal Error Rate (EER %)")
    plt.title("Speaker Verification Performance (Lower is Better)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("results/performance_comparison.png")

    # --- Metrics Table ---
    with open("results/metrics_table.md", "w") as f:
        f.write("# Evaluation Metrics (LibriSpeech Full)\n\n")
        f.write("| Model Mode | EER (%) | Performance Note |\n")
        f.write("| :--- | :--- | :--- |\n")
        for mode, eer in results.items():
            note = "Target Metric" if mode == "disentangled" else "Baseline"
            if mode == "improved": note = "Proposed Attention Improvement"
            f.write(f"| {mode} | {eer:.2f}% | {note} |\n")

if __name__ == "__main__":
    evaluate()