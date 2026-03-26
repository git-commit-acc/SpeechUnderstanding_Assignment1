# Q2: Speaker Recognition using Disentangled Representation Learning

## 📌 Overview

This project implements a simplified version of the paper
**"Disentangled Representation Learning for Environment-agnostic Speaker Recognition"**.

The goal is to improve speaker verification performance under noisy conditions by separating speaker identity from environmental noise.

---

## 📂 Folder Structure

```
q2/
│── train.py
│── eval.py
│── configs/
│── results/
│   ├── results_table.csv
│   ├── performance_comparison.png
│── review.pdf
│── q2_readme.md
```

---

## ⚙️ Setup Instructions

1. Create environment:

```
conda create -n speaker python=3.10
conda activate speaker
```

2. Install dependencies:

```
pip install torch torchaudio numpy pandas matplotlib scikit-learn
```

---

## 🚀 Training

Run baseline model:

```
python train.py --config configs/baseline.yaml
```

Run disentangled model:

```
python train.py --config configs/disentangled.yaml
```

Run improved model:

```
python train.py --config configs/improved.yaml
```

---

## 📊 Evaluation

```
python eval.py --model checkpoints/model.pth
```

Metrics used:

* Equal Error Rate (EER)
* Verification Accuracy

---

## 📈 Results

| Model        | EER (%) |
| ------------ | ------- |
| Baseline     | 13.4    |
| Disentangled | 12.9    |
| Improved     | 10.8    |

The disentangled approach improves performance by making embeddings more robust to noise. The proposed improvement further reduces error.

---

## 📉 Visualization

The results plot (`performance_comparison.png`) shows that:

* Disentangled model performs better than baseline
* Improved model gives the lowest EER

---

## 💡 Proposed Improvement

We introduced a **Lombard-aware attention mechanism**:

* Uses environment embedding to adapt speaker features
* Instead of removing noise completely, it learns how noise affects speech

This improves performance in noisy conditions.

---
