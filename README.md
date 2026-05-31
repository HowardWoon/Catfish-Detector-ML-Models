<div align="center">
  
  # 🕵️ Catfish Detector AI
  **WIA1006 Machine Learning Project • Group 7**
  
  Detecting Fake Personalities Through Advanced Behavioral Intelligence, Mathematical Anomaly Detection, and Ensemble Machine Learning.
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DygpqueWNxma9PUHvAIp2jaK737kO7kZ)
</div>

---

## 📌 Project Overview
Romance scams and catfishing have become billion-dollar problems on modern dating platforms. The **Catfish Detector AI** is an advanced machine learning pipeline and interactive web application designed to automatically identify fraudulent profiles. 

Rather than relying purely on easily-faked texts, natural language processing, or stolen images (which can easily bypass traditional filters), our AI focuses on **mathematical behavioral patterns**. By analyzing the raw telemetry of how a user physically interacts with the platform—such as swiping speeds, message density ratios, and bio-length efficiencies—our system can definitively detect automated bots, romance scammers, and fake profiles with extreme accuracy.

---

## 🔬 Theoretical Framework & Feature Engineering
Raw data alone is not enough to catch sophisticated scammers. Our system engineers 51 complex mathematical features to expose unnatural behavior. Some of our core theoretical heuristics include:

* **Engagement Score** `[messages_sent / (app_usage_time + 1)]`: Scammers often blast hundreds of messages in a very short time window. A high engagement score combined with low usage time is a massive statistical anomaly.
* **Swipe Intensity** `[swipe_right_ratio / (app_usage_time + ε)]`: Bots are designed to indiscriminately swipe right on every profile instantly to maximize match yields.
* **Bio Efficiency** `[bio_length / (messages_sent + 1)]`: Authentic users typically write longer, thoughtful bios. Fraudsters often have extremely short, generic bios but incredibly high message outputs.
* **Match Conversion Rate**: Scammers often receive a disproportionately high number of "Likes" relative to their actual "Mutual Matches", indicating spam-like behavior that real users are swiping left on.

---

## 🧠 The Machine Learning Pipeline (V32 ULTIMATE)
Our ultimate **V32** pipeline represents the pinnacle of our mathematical optimization, completely eliminating the "black box" nature of traditional AI:

### 1. Data Cleaning & Outlier Mitigation
Before training, we deploy a **RobustScaler** to normalize features using statistics that are robust to extreme outliers (using the interquartile range). We strictly drop highly correlated features (Pearson correlation > 0.95) and apply a **VarianceThreshold** to remove constants, ensuring the AI only learns from statistically significant data.

### 2. Algorithmic Data Balancing (SMOTE-Tomek)
Because dating apps are heavily imbalanced (e.g., 93% Genuine users, 7% Scammers), standard AI models will simply guess "Genuine" to achieve 93% accuracy. We solve this using **SMOTE-Tomek**:
* **SMOTE (Synthetic Minority Over-sampling Technique)** mathematically generates new, realistic scammer profiles by interpolating between existing minority data points in N-dimensional space.
* **Tomek Links** simultaneously identifies and removes overlapping, noisy samples on the decision boundary to create clean, mathematically separable classes.

### 3. Dynamic Principal Component Analysis (PCA)
For deep learning models like our Neural Network, we apply PCA to compress the dataset while strictly retaining **95% of the mathematical variance**. This dramatically reduces computational overhead and filters out microscopic statistical noise, allowing the Neural Network to train faster and converge more accurately.

### 4. The 6-Model Ensemble Engine
No single algorithm is perfect. We utilize a highly diverse ensemble of 6 distinct algorithms, dynamically tuned using `RandomizedSearchCV` across a Stratified K-Fold cross-validation:
1. **XGBoost & Random Forest**: Tree-based heavyweights that excel at finding highly complex, non-linear behavioral correlations.
2. **Extra Trees**: Utilizes highly randomized decision splits to forcefully prevent overfitting on the synthetic data.
3. **Decision Tree**: Provides a fully transparent, interpretable mathematical flowchart that humans can read.
4. **Logistic Regression**: Establishes a solid, statistically interpretable baseline using sigmoid probabilities.
5. **Multi-Layer Perceptron (Neural Network)**: Detects deep, hidden correlations in the latent space.

### 5. Explainable AI (SHAP)
We utilize **SHapley Additive exPlanations (SHAP)** based on cooperative game theory. Instead of the AI just saying "This is a Catfish", SHAP calculates the exact marginal contribution of every single behavioral feature. It visually breaks down *why* a profile was flagged, allowing security teams to audit the AI's exact "thought process".

---

## 🛡️ The Blended Threat Detection Engine
When a profile is scanned on our live web app, it goes through a dual-layered security protocol:

1. **The Machine Learning Vote**: The 6 models independently evaluate the profile and cast a vote based on heavily optimized probability thresholds (ranging from 0.10 to 0.90 to maximize F1-Scores).
2. **The Heuristic Fallback Engine (Z-Score Anomaly Detection)**: The raw input is compared against the population mean using Z-Scores. If a user is doing something physically impossible (e.g., 4 Standard Deviations above the mean for swiping speed), the heuristic engine generates an extreme risk score.

**The Final Verdict:** The system dynamically blends the ML probabilities with the Heuristic risk score. If the ML models are extremely confident OR the heuristic rules detect a physically impossible anomaly, the profile is immediately flagged as a **CATFISH**.

---

## 💻 Running the Live Web Application

The project includes a fully functional, highly polished local web server with a modern glassmorphism UI. We have **pre-compiled all 6 Machine Learning models** (a 58MB bundle) and included them in the repository. **You can launch the website instantly with zero training time!**

### Prerequisites
Ensure you have Python 3.9+ installed, then set up the environment:
```bash
# 1. Clone the repository
git clone https://github.com/HowardWoon/Catfish-Detector-ML-Models.git
cd Catfish-Detector-ML-Models

# 2. Install the required dependencies
pip install -r requirements.txt
```

### Launching the App
Simply run the master launcher script in your terminal:
```bash
python run_web.py
```
*Note: Do not run `app.py` directly. The `run_web.py` script is our official launcher. It will dynamically load the AI brain, boot the local Flask server, wait for a successful connection, and then automatically open your web browser to `http://127.0.0.1:5000/`.*

---

## 📂 Repository Structure & Architecture
* `/WIA1006_Catfish_Group7_V32_Ultimate.ipynb` - The primary, fully-documented ML pipeline Colab notebook containing all theoretical breakdowns.
* `/dating_app_behavior_dataset.csv` - The core dataset containing 50,000 raw behavioral profiles.
* `/run_web.py` - Master Python script used to launch the interactive website safely.
* `/catfish_core.py` - The core backend engine handling the Machine Learning pipeline, PCA transformations, feature engineering, and the heuristic scoring logic.
* `/webapp/app.py` - Flask server handling all backend routing, JSON endpoints, and API logic.
* `/webapp/validation.py` - Automated integration testing to ensure the ML models boot correctly on the web server.
* `/webapp/templates/` - HTML files housing the frontend UI for the interactive Web Scanner, Explainability Center, and Model Battle Arena.
* `/webapp/static/` - Custom CSS stylesheets, SVGs, and dynamic frontend JavaScript.
* `/artifacts/` - Houses the pre-compiled `detector_bundle.pkl` (the trained AI models) and auto-generated matplotlib diagnostic plots.

---
<div align="center">
  <i>Developed with ❤️ for WIA1006 Machine Learning.</i>
</div>
