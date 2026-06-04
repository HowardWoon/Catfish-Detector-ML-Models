<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=CATFISH%20DETECTOR%20AI&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=WIA1006%20Machine%20Learning%20•%20Group%207&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1AR7Mv0Eg1iGw2IWA1pB_Xt9RZHPHeLCx#scrollTo=ce2f5a66)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Server-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **Detecting Fake Personalities Through Advanced Behavioral Intelligence,**  
> **Mathematical Anomaly Detection, and Ensemble Machine Learning.**

<br/>

</div>

---

## 🌟 Try Our Live AI Notebook!
> ### 👉 [Click Here to Open Ultimate Pipeline in Google Colab!](https://colab.research.google.com/drive/1AR7Mv0Eg1iGw2IWA1pB_Xt9RZHPHeLCx#scrollTo=ce2f5a66) 👈
> **This is the core of our project.** Run the complete 6-Model Ensemble pipeline, visualize our mathematical behavioral data, and execute the final machine learning training process directly in your browser with zero setup required!
>
> **V32.0 Bulletproof Edition:** Cells **12 → 12a–12f → 12g** train all six models with `RandomizedSearchCV` for extreme hyperparameter tuning, Colab-aware parallel CV (`n_jobs=-1`), reproducible subsampling, and a validation gate in **12g** before metrics/SHAP cells.

---

## ⚡ At a Glance

| Stat | Value |
|------|-------|
| 🧬 Total Features Engineered | **51 behavioral features** |
| 🤖 Models in Ensemble | **6 algorithms (Week 7-9 Aligned)** |
| 📊 Training Profiles | **50,000 behavioral records** |
| ⚖️ Class Imbalance Handled | **SMOTE-Tomek resampling** |
| 🗜️ Model Bundle Size | **58 MB (pre-compiled)** |
| 🚀 Startup Time | **Zero training — instant launch** |

---

## 🎯 The Problem

Romance scams and catfishing are **billion-dollar crises** on modern dating platforms. Traditional defenses fail because:

- 🖼️ **Stolen images** bypass visual filters trivially
- 💬 **NLP-based checks** are fooled by AI-generated text
- 📝 **Manual reports** are too slow and reactive

**Our approach is different.** We don't read what a user *says* — we analyze the mathematical fingerprint of *how they behave*.

> A scammer can steal a photo. A scammer can paste fake text. But a scammer **cannot** fake the physics of genuine human interaction patterns.

---

## 🔬 Feature Engineering — 51 Behavioral Signals

Raw data cannot catch sophisticated scammers. Our pipeline engineers **51 complex mathematical features** that expose unnatural behavior invisible to the human eye.

<details>
<summary><b>🧮 Core Heuristic Features (click to expand)</b></summary>

<br/>

| Feature | Formula | What It Catches |
|---------|---------|-----------------|
| **Engagement Score** | `messages_sent / (app_usage_time + 1)` | Message-blasting bots operating in burst windows |
| **Swipe Intensity** | `swipe_right_ratio / (app_usage_time + ε)` | Indiscriminate auto-swiping to farm matches |
| **Bio Efficiency** | `bio_length / (messages_sent + 1)` | Short generic bios paired with abnormal message output |
| **Match Conversion Rate** | `likes_received / mutual_matches` | Spam-like profiles that real users reject at high rates |

</details>

---

## 🧠 ML Pipeline — ULTIMATE

```
Raw Data → Cleaning → Balancing → PCA → 6-Model Ensemble → SHAP → Verdict
```

### Step 1 · Data Cleaning & Outlier Mitigation

```python
RobustScaler()          # Normalizes using IQR — immune to extreme outliers
Pearson correlation > 0.95  # Drops redundant features automatically
VarianceThreshold()     # Eliminates zero-variance constants
```

### Step 2 · Algorithmic Class Balancing (SMOTE-Tomek)

Real dating app data is brutally imbalanced (93% Genuine / 7% Scammer). A naive model exploits this and achieves 93% accuracy by guessing "Genuine" every time. We solve this with a two-phase resampling attack:

```
SMOTE  → Synthesizes realistic new scammer profiles by interpolating in N-dimensional space
Tomek  → Removes noisy boundary samples that blur the decision boundary
```

### Step 3 · Dynamic PCA

For deep learning components, PCA compresses the feature space while **strictly retaining 95% of mathematical variance** — reducing noise and dramatically accelerating neural convergence.

### Step 4 · 6-Model Ensemble Engine

Each model is independently tuned via `RandomizedSearchCV` across stratified K-Fold splits.

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE VOTE                            │
│                                                             │
│  ① Gaussian Mixture → Probabilistic distribution (Week 9)   │
│  ② KMeans           → Label-aware clustering (Week 8)       │
│  ③ Support Vector   → Hyperplane margin isolation (Week 9)  │
│  ④ Decision Tree    → Human-readable audit trail            │
│  ⑤ Logistic Reg.    → Sigmoid probability baseline          │
│  ⑥ Neural Network   → Latent space pattern detection        │
│                                                             │
│         Dynamic Threshold Range: 0.10 → 0.90                │
│         (Per-model F1-Score optimized)                      │
└─────────────────────────────────────────────────────────────┘
```

### Step 5 · Explainable AI (SHAP)

The system never just says *"This is a Catfish."* Using **SHapley Additive exPlanations** (rooted in cooperative game theory), every verdict is broken down into the exact marginal contribution of each behavioral feature — producing a fully auditable AI thought process.

---

## 🛡️ Blended Threat Detection Engine

When a profile is submitted to the live scanner, it passes through **two independent security layers** before a verdict is issued:

```
Profile Input
     │
     ├──► [Layer 1] ML Ensemble Vote
     │         6 models score independently
     │         Weighted probability fusion
     │
     └──► [Layer 2] Z-Score Heuristic Engine
               Flags physically impossible behavior
               (e.g., swipe speed > 4σ from population mean)
                          │
                          ▼
              ┌───────────────────────┐
              │   DYNAMIC BLENDING    │
              │   ML Prob + Z-Score   │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
           GENUINE               ⚠ CATFISH
```

> **Rule:** If the ML ensemble is extremely confident **OR** the heuristic engine detects a physically impossible anomaly — the profile is immediately flagged.

---

## 💻 Two-Pronged Web Interface

We engineered two distinct web interfaces to demonstrate the AI:

1. **Live Profile Scanner (Data Science Tool):** Input exact raw numerical vectors (e.g., exactly 893 likes, 14 matches) into the core ML backend to manually analyze model probabilities and heuristic anomaly z-scores.
2. **Profile Risk Simulator (Frontend Simulator):** Build hypothetical dating profiles using plain English ("Rarely online", "Extremely chatty"). The interface automatically translates human behavior into mathematical vectors and runs the Safety Report.

---

## 💻 Quick Start

### Prerequisites

```bash
Python 3.9+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/HowardWoon/Catfish-Detector-ML-Models.git
cd Catfish-Detector-ML-Models

# Install dependencies
pip install -r requirements.txt
```

### Launch

```bash
python run_web.py
```

> ⚠️ **Do not run `app.py` directly.** `run_web.py` is the official launcher — it boots the AI brain, initializes the Flask server, validates model integrity, and opens your browser to `http://127.0.0.1:5000/` automatically.

---

## 📂 Repository Structure

```
Catfish-Detector-ML-Models/
│
├── 📓 WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb  ← Full ML pipeline & theory
├── 📊 dating_app_behavior_dataset.csv             ← 50,000 behavioral profiles
├── 🚀 run_web.py                                  ← Official launcher (use this)
├── ⚙️  catfish_core.py                            ← Core ML engine & heuristics
│
├── webapp/
│   ├── 🌐 app.py                                  ← Flask server & API routing
│   ├── 🧪 validation.py                           ← Boot-time integration tests
│   ├── templates/                                 ← Web Scanner · Explainability · Arena
│   └── static/                                   ← CSS · JS · SVG assets
│
└── artifacts/
    ├── 🤖 detector_bundle.pkl                     ← Pre-compiled 58MB model bundle
    └── 📈 *.png                                   ← Auto-generated diagnostic plots
```

---

## 🏗️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=flat-square)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-ec4899?style=flat-square)

</div>

---

<div align="center">

**WIA1006 Machine Learning · Group 7 · Universiti Malaya**

*Built with ❤️ — detecting deception, one behavioral signature at a time.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

</div>
