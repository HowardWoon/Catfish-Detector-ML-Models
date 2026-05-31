<div align="center">
  
  # Catfish Detector AI
  **WIA1006 Machine Learning Project • Group 7**
  
  Detecting Fake Personalities Through Behavioral Intelligence.
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HowardWoon/Catfish-Detector-ML-Models/blob/main/WIA1006_Catfish_Group7_V32_Ultimate.ipynb)
</div>

---

## 📌 Project Overview
The **Catfish Detector AI** is an advanced machine learning pipeline and interactive web application designed to identify fraudulent "Catfish" profiles on dating platforms. Rather than relying purely on text or image analysis, our AI detects subtle, high-risk **behavioral patterns**—such as swipe-to-message ratios, engagement intensity, and suspicious bio efficiencies.

## 🚀 Key Features
* **6-Model Ensemble Engine**: Analyzes profiles simultaneously using XGBoost, Random Forest, Extra Trees, Decision Tree, Logistic Regression, and a Multi-Layer Perceptron (Neural Network).
* **Robust Behavioral Analytics**: Extracts complex behavioral mathematical features from raw interaction data.
* **Explainable AI (SHAP)**: Uses SHapley Additive exPlanations to visually break down exactly *why* a profile was flagged as a catfish, removing the "black box" of machine learning.
* **Interactive Live Scanner**: A stunning, real-time web UI that allows users to adjust behavioral sliders and watch the AI models vote on the threat level instantly.

## 🧠 The Machine Learning Pipeline (V32 ULTIMATE)
Our ultimate **V32** Notebook represents the pinnacle of our optimization:
1. **Dataset Generation**: 50,000 rows of dating app behavior.
2. **Feature Engineering**: Calculates `engagement_score`, `swipe_intensity`, `msg_per_minute`, etc.
3. **Data Balancing**: Utilizes **SMOTE-Tomek** to mathematically balance the heavily skewed dataset.
4. **Dynamic PCA Integration**: Retains 95% of mathematical variance to optimize Neural Network performance.
5. **Hyperparameter Tuning**: Dynamically tuned using `RandomizedSearchCV` to maximize F1-Scores.
6. **Threshold Optimization**: Dynamically searches across a wide range of probability thresholds to perfectly balance precision and recall.

## 📈 Performance & Evaluation
Our ensemble models achieve state-of-the-art accuracy:
* **F1-Scores**: ~99% (XGBoost & Random Forest)
* **Diagnostics**: Evaluated using comprehensive PR Curves, ROC-AUC Curves, and Probability Distributions.

## 💻 Running the Live Web Application

The project includes a fully functional, highly polished local web server. We have **pre-compiled all 6 Machine Learning models** (58MB) and included them in the repository, meaning you can launch the website instantly with zero training time!

### Prerequisites
Make sure you have Python installed, then run:
```bash
git clone https://github.com/HowardWoon/Catfish-Detector-ML-Models.git
cd Catfish-Detector-ML-Models
pip install -r requirements.txt
```

### Launching the App
Simply run the master launcher script:
```bash
python run_web.py
```
*Note: Do not run `app.py` directly. `run_web.py` is the official launcher that automatically monitors the server and safely opens your web browser to `http://127.0.0.1:5000/` once the AI is ready!*

## 📂 Repository Structure
* `/WIA1006_Catfish_Group7_V32_Ultimate.ipynb` - The primary, fully-documented ML pipeline notebook.
* `/dating_app_behavior_dataset.csv` - The injected dataset.
* `/run_web.py` - Master launcher script for the interactive website.
* `/catfish_core.py` - Core machine learning logic and dynamic artifact compiler.
* `/webapp/app.py` - Flask server handling the backend routing and API endpoints.
* `/webapp/templates/` - HTML files for the interactive web scanner.
* `/webapp/static/` - CSS stylesheets and JavaScript.
* `/artifacts/` - Pre-compiled `detector_bundle.pkl` containing the trained AI models.

---
<div align="center">
  <i>Developed with ❤️ for WIA1006 Machine Learning.</i>
</div>
