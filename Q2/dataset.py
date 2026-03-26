import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, feature_dim=40):
        self.root_dir = root_dir
        self.samples = []
        # Walk through LibriSpeech structure: root/speaker_id/chapter_id/*.flac
        for speaker_id in sorted(os.listdir(root_dir)):
            speaker_path = os.path.join(root_dir, speaker_id)
            if os.path.isdir(speaker_path):
                for chapter_id in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter_id)
                    for file in os.listdir(chapter_path):
                        if file.endswith(".flac"):
                            self.samples.append({
                                "path": os.path.join(chapter_path, file),
                                "speaker_id": int(speaker_id)
                            })
        
        # Map speaker_ids to continuous range 0 -> N
        df = pd.DataFrame(self.samples)
        self.id_map = {old: new for new, old in enumerate(df['speaker_id'].unique())}
        self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=feature_dim)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, _ = torchaudio.load(sample["path"])
        mel = self.feature_transform(waveform).squeeze(0) # [Freq, Time]
        
        # Pad or truncate to a fixed length (e.g., 200 frames)
        if mel.shape[1] < 200:
            mel = torch.nn.functional.pad(mel, (0, 200 - mel.shape[1]))
        else:
            mel = mel[:, :200]
            
        label = self.id_map[sample["speaker_id"]]
        # Using Chapter ID as a proxy for 'Environment' for disentanglement
        env_label = int(sample["path"].split(os.sep)[-2]) % 5 
        
        return mel, label, env_label