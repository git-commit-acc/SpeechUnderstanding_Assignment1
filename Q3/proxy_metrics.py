import torch

def audio_acceptability_check(original, transformed):
    """
    Proxy for DNSMOS/FAD.
    Measures Mean Squared Error to ensure the audio isn't 
    destroyed (Signal Acceptability).
    """
    # Ensure same length
    min_len = min(original.shape[1], transformed.shape[1])
    mse = torch.mean((original[:, :min_len] - transformed[:, :min_len])**2)
    
    if mse > 0.1:
        print("Warning: High Artifacts detected (Toxicity Trap).")
    else:
        print(f"Audio Acceptability Score (MSE): {1 - mse:.4f}")