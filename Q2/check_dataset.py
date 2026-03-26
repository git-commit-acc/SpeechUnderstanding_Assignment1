# Save as check_system.py
import torch
from dataset import LibriSpeechDataset

dataset = LibriSpeechDataset(r"D:\Projects\SpeechUnderstanding\M25DE1035\Q2\data\LibriSpeech\train-clean-100")
print(f"Total files found: {len(dataset)}")

# Try to load just ONE item
try:
    sample, label, env = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print("Data loading is working fine.")
except Exception as e:
    print(f"Error loading data: {e}")

print(f"Is CUDA available? {torch.cuda.is_available()}")