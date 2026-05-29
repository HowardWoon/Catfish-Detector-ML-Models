<div align="center">
  <img src="https://raw.githubusercontent.com/HowardWoon/Catfish-Detector-ML-Models/main/webapp/static/cyber-eye.svg" width="100" alt="Cyber Eye Logo">
  
  # Catfish Detector AI
  **WIA1006 Machine Learning Project • Group 7**
  
  Detecting Fake Personalities Through Behavioral Intelligence.
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HowardWoon/Catfish-Detector-ML-Models/blob/main/WIA1006_Catfish_Group7_V19_CHAMPION.ipynb)
</div>

---

## 📌 Project Overview
The **Catfish Detector AI** is an advanced machine learning pipeline and interactive web application designed to identify fraudulent "Catfish" profiles on dating platforms. Rather than relying purely on text or image analysis, our AI detects subtle, high-risk **behavioral patterns**—such as swipe-to-message ratios, engagement intensity, and suspicious bio efficiencies.

## 🚀 Key Features
* **6-Model Ensemble Engine**: Analyzes profiles simultaneously using XGBoost, Random Forest, Extra Trees, Decision Tree, Logistic Regression, and a Multi-Layer Perceptron (Neural Network).
* **Robust Behavioral Analytics**: Extracts 51 complex features from raw interaction data.
* **Explainable AI (SHAP)**: Uses SHapley Additive exPlanations to visually break down exactly *why* a profile was flagged as a catfish, removing the "black box" of machine learning.
* **Interactive Live Scanner**: A stunning, real-time web UI that allows users to adjust behavioral sliders and watch the AI models vote on the threat level instantly.

## 🧠 The Machine Learning Pipeline (V19 CHAMPION)
Our ultimate [V19 Champion Notebook](WIA1006_Catfish_Group7_V19_CHAMPION.ipynb) represents the pinnacle of our optimization:
1. **Dataset Generation**: 50,000 rows of synthetic dating app behavior.
2. **Feature Engineering**: Calculates `engagement_score`, `swipe_intensity`, `msg_per_minute`, etc.
3. **Data Balancing**: Utilizes **SMOTE-Tomek** to handle class imbalance, synthetically generating realistic minority class samples while pruning ambiguous Tomek links.
4. **Training without PCA**: Models train natively on the full 51 features for maximum accuracy without lossy compression.
5. **Hyperparameter Tuning**: Dynamically tuned using `RandomizedSearchCV`.

## 📈 Performance & Evaluation
Our ensemble models achieve state-of-the-art accuracy:
* **F1-Scores**: ~99% (XGBoost & Random Forest)
* **Diagnostics**: Evaluated using comprehensive ROC-AUC Curves and Calibration (Reliability) Diagrams.

## 💻 Running the Live Web Application

The project includes a fully functional, highly polished local web server with a modern glassmorphism UI.

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install flask scikit-learn pandas numpy xgboost
```

### Launching the App
1. Clone this repository to your local machine.
2. Ensure you have run the Colab Notebook to generate the `detector_bundle.pkl` model asset, and place it in the `webapp/models/` directory (or use the one provided).
3. Start the Flask server:
```bash
python app.py
```
4. Open your browser and navigate to: `http://127.0.0.1:5000/`

## 📂 Repository Structure
* `/WIA1006_Catfish_Group7_V19_CHAMPION.ipynb` - The primary ML pipeline notebook.
* `/dating_app_behavior_dataset.csv` - The injected synthetic dataset.
* `/app.py` - Flask server handling the backend logic for the web UI.
* `/catfish_core.py` - Core machine learning logic for local builds.
* `/webapp/templates/` - HTML files for the interactive web scanner.
* `/webapp/static/` - CSS stylesheets, SVGs, and JavaScript.

---
<div align="center">
  <i>Developed with ❤️ for WIA1006 Machine Learning.</i>
</div>
