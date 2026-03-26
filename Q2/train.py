import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import LibriSpeechDataset
from models.disentangler import SpeakerNet
import json
import os
import random
import numpy as np

# Fix seeds so results are reproducible across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(config, mode):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = LibriSpeechDataset(config["data_path"], config["feature_dim"])
    print(f"Total samples found: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No data found. Check your config.json path.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = SpeakerNet(config, mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for i, (batch_x, batch_y_spk, batch_y_env) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y_spk = batch_y_spk.to(device)
            batch_y_env = batch_y_env.to(device)

            optimizer.zero_grad()
            spk_logits, env_logits, spk_emb, env_emb = model(batch_x)

            l_spk = F.cross_entropy(spk_logits, batch_y_spk)

            if mode != "baseline":
                l_env = F.cross_entropy(env_logits, batch_y_env)
                l_ortho = torch.mean(torch.abs(F.cosine_similarity(spk_emb, env_emb)))
                loss = l_spk + l_env + config["ortho_penalty_weight"] * l_ortho
            else:
                loss = l_spk

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Mode: {mode} | Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"  --> Epoch {epoch} done. Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()}")
        scheduler.step()

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{mode}.pt")
        print(f"  Checkpoint saved: checkpoints/{mode}.pt")


if __name__ == "__main__":
    with open("configs/config.json", "r") as f:
        conf = json.load(f)

    # Retrain all three with fixed seed for fair comparison
    train_model(conf, "baseline")
    train_model(conf, "disentangled")
    train_model(conf, "improved")