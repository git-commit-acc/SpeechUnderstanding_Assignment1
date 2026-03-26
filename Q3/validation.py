import torch
import torchaudio
import numpy as np

def compute_band_energy(spec, sr, n_fft):
    """
    Compute energy in speech band (300Hz - 8000Hz)
    """

    # Correct frequency bins for STFT
    freqs = torch.fft.rfftfreq(n_fft, d=1/sr)  # ✅ FIX

    mask = (freqs >= 300) & (freqs <= 8000)

    # spec shape: [channel, freq, time]
    energy = torch.abs(spec) ** 2

    # Apply mask on frequency dimension
    band_energy = energy[:, mask, :].mean()
    total_energy = energy.mean()

    return (band_energy / total_energy).item()


def check_acceptability(orig_path, obf_path):
    orig, sr1 = torchaudio.load(orig_path)
    obf, sr2 = torchaudio.load(obf_path)

    assert sr1 == sr2, "Sampling rates must match!"

    # Normalize audio (important!)
    orig = orig / (orig.abs().max() + 1e-8)
    obf = obf / (obf.abs().max() + 1e-8)

    # Match lengths
    min_len = min(orig.shape[1], obf.shape[1])
    orig = orig[:, :min_len]
    obf = obf[:, :min_len]

    # 1. Signal distortion (MSE)
    mse = torch.mean((orig - obf) ** 2).item()

    # 2. Spectral consistency
    n_fft = 512
    window = torch.hann_window(n_fft)

    spec_orig = torch.stft(orig, n_fft=n_fft, window=window, return_complex=True)
    spec_obf = torch.stft(obf, n_fft=n_fft, window=window, return_complex=True)

    band_orig = compute_band_energy(spec_orig, sr1, n_fft)
    band_obf = compute_band_energy(spec_obf, sr1, n_fft)

    spectral_diff = abs(band_orig - band_obf)

    # 3. SNR (better perceptual proxy)
    noise = orig - obf
    snr = 10 * torch.log10(
        torch.mean(orig ** 2) / (torch.mean(noise ** 2) + 1e-8)
    ).item()

    # Final acceptability score
    score = 1 - (0.5 * mse + 0.3 * spectral_diff + 0.2 * max(0, -snr / 20))

    print("\n--- Validation Report ---")
    print(f"MSE Distortion        : {mse:.4f}")
    print(f"Spectral Shift        : {spectral_diff:.4f}")
    print(f"SNR (dB)              : {snr:.2f}")
    print(f"Acceptability Score   : {score:.2f}/1.00")

    if score < 0.6:
        print("RESULT: Toxicity Trap Detected (Too much distortion)")
    else:
        print("RESULT: Acceptable Audio Quality")


if __name__ == "__main__":
    check_acceptability(
        "q3/examples/original.wav",
        "q3/examples/obfuscated.wav"
    )