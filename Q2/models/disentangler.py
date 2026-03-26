import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class StatsPool(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1) + 1e-6
        return torch.cat([mean, std], dim=-1)


class SpeakerNet(nn.Module):
    def __init__(self, config, mode="baseline"):
        super().__init__()
        self.mode = mode
        feat_dim = config["feature_dim"]
        latent = config["latent_dim"]

        self.encoder = nn.Sequential(
            ConvBlock(feat_dim, 128, kernel=5),
            ConvBlock(128, 128, kernel=3, dilation=2),
            ConvBlock(128, 256, kernel=3, dilation=4),
            ConvBlock(256, 256, kernel=3),
        )

        self.pool = StatsPool()
        encoder_out_dim = 256 * 2

        self.speaker_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent)
        )
        self.spk_classifier = nn.Linear(latent, config["num_speakers"])

        if self.mode in ["disentangled", "improved"]:
            self.env_head = nn.Sequential(
                nn.Linear(encoder_out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent)
            )
            self.env_classifier = nn.Linear(latent, config["num_environments"])

        if self.mode == "improved":
            # Proposed improvement: an instance normalization layer on the speaker
            # embedding before classification. The idea is that channel-wise mean
            # and variance in the embedding often carry environment/channel info
            # (recording conditions, microphone, room acoustics). By normalizing
            # these out we make the embedding more environment-invariant without
            # relying on the quality of the env labels at all.
            # We use a learnable affine transform so the network can undo the
            # normalization if a dimension is genuinely speaker-discriminative.
            self.spk_norm = nn.LayerNorm(latent, elementwise_affine=True)

    def forward(self, x):
        features = self.encoder(x)
        pooled = self.pool(features)

        spk_emb = self.speaker_head(pooled)

        if self.mode == "baseline":
            return self.spk_classifier(spk_emb), None, spk_emb, None

        env_emb = self.env_head(pooled)
        env_logits = self.env_classifier(env_emb)

        if self.mode == "improved":
            # Normalize speaker embedding across the latent dimension.
            # This removes global mean/variance shifts caused by environment,
            # which is the main failure mode our critique identified.
            spk_emb = self.spk_norm(spk_emb)

        return self.spk_classifier(spk_emb), env_logits, spk_emb, env_emb