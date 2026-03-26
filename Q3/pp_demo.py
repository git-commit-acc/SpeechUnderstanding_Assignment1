import os
import torchaudio
from privacymodule import PrivacyModule

def run_demo(input_audio_file):
    # Ensure the output directory exists
    os.makedirs("q3/examples", exist_ok=True)
    
    # Initialize the privacy module
    pm = PrivacyModule()
    
    print(f"Loading audio file: {input_audio_file}")
    waveform, sr = torchaudio.load(input_audio_file)
    
    # Apply the Privacy Preserving transformation (Obfuscation)
    # shift_factor > 1.0 shifts pitch up (e.g., Male to Female/Younger)
    pm.n_steps = 4.0  # control pitch shift (or compute from factor)
    protected_audio = pm(waveform)
    
    # Save the pairs for your audit deliverables
    torchaudio.save("q3/examples/original.wav", waveform, sr)
    torchaudio.save("q3/examples/obfuscated.wav", protected_audio, sr)
    
    print("Success! Biometric obfuscation complete.")
    print("Files saved in: q3/examples/")

if __name__ == "__main__":
    # REPLACE THIS PATH with a real .flac file from your dataset
    # Example: Speaker 19 (Kara Shallenberg) from your SPEAKERS.TXT
    target_audio = r"D:\Projects\SpeechUnderstanding\M25DE1035\Assignment1\Q3\data\27-123349-0013.flac"
    
    if os.path.exists(target_audio):
        run_demo(target_audio)
    else:
        print(f"Error: Could not find the audio file at {target_audio}")
        print("Please verify the path to a valid .flac file.")
