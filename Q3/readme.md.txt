# Q3: Ethical Auditing & Privacy-Preserving Audio System

## 📌 Overview

This project implements:

* Bias auditing on LibriSpeech dataset
* Privacy-preserving voice transformation (PyTorch)
* Fairness-aware training loss
* Audio validation using proxy metrics (MSE, Spectral Consistency, SNR)

---

## 📁 Folder Structure

```
q3/
│
├── audit.py
├── privacymodule.py
├── pp_demo.py
├── train_fair.py
├── validation.py
├── proxy_metrics.py
│
├── examples/
│   ├── original.wav
│   └── obfuscated.wav
│
├── audit_plots.pdf
└── q3_report.pdf
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchaudio pandas matplotlib numpy
```

---

## 📊 Step 1: Run Bias Audit

### Command:

```bash
python audit.py
```

### What it does:

* Reads LibriSpeech metadata (`SPEAKERS.TXT`)
* Computes:

  * Gender distribution
  * Duration imbalance
* Generates:

  * `q3/audit_plots.pdf`

### ⚠️ Important:

Update dataset path inside `audit.py`:

```python
path = "your_path_to/SPEAKERS.TXT"
```

---

## 🔐 Step 2: Run Privacy-Preserving Transformation

### Command:

```bash
python pp_demo.py
```

### What it does:

* Loads a speech file
* Applies pitch shift + noise
* Saves:

  * `original.wav`
  * `obfuscated.wav`

### ⚠️ Important:

Update audio path in `pp_demo.py`:

```python
target_audio = "your_audio_file.flac"
```

---

## 📈 Step 3: Validate Audio Quality

### Command:

```bash
python validation.py
``
```
