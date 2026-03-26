import torch
import torch.nn as nn
import torchaudio.functional as F

class PrivacyModule(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, waveform):
        """
        Default forward (used when calling pm(waveform))
        """
        return self.transform(waveform, shift_factor=1.2)

    def transform(self, waveform, shift_factor=1.2):
        """
        Privacy-preserving transformation
        """

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert factor → semitones
        n_steps = (shift_factor - 1.0) * 12
        n_steps = max(min(n_steps, 6), -6)

        # Pitch shift
        anonymized = F.pitch_shift(
            waveform,
            self.sample_rate,
            n_steps=n_steps
        )

        # Add noise
        noise = torch.randn_like(anonymized) * 0.003
        anonymized = anonymized + noise

        # Clamp audio
        anonymized = torch.clamp(anonymized, -1.0, 1.0)

        return anonymized