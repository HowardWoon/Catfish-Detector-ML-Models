<div align="center">
  
  # 🕵️ Catfish Detector AI
  **WIA1006 Machine Learning Project • Group 7**
  
  Detecting Fake Personalities Through Behavioral Intelligence.
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DygpqueWNxma9PUHvAIp2jaK737kO7kZ)
</div>

---

## 📌 Project Overview
Romance scams and catfishing have become billion-dollar problems on modern dating platforms. The **Catfish Detector AI** is an advanced machine learning pipeline and interactive web application designed to automatically identify fraudulent profiles. 

Rather than relying purely on easily-faked texts or stolen images, our AI analyzes **subtle, high-risk behavioral patterns**. By looking at how a user mathematically interacts with the platform (e.g., swiping speeds, message densities, and bio-length efficiencies), our system can detect automated bots, romance scammers, and fake profiles with extreme accuracy.

## 🚀 Key Features
* **6-Model Ensemble Engine**: Analyzes profiles simultaneously using a diverse array of algorithms:
  * **XGBoost & Random Forest**: Tree-based heavyweights for non-linear pattern recognition.
  * **Extra Trees**: Highly randomized trees to prevent overfitting.
  * **Decision Tree**: Provides a fully transparent, interpretable mathematical flowchart.
  * **Logistic Regression**: Establishes a solid statistical baseline.
  * **Multi-Layer Perceptron (Neural Network)**: Detects deep, hidden correlations in the data.
* **Robust Behavioral Analytics**: Extracts complex mathematical features (like `swipe_intensity` and `engagement_density`) from raw interaction data.
* **Explainable AI (SHAP)**: Uses SHapley Additive exPlanations to visually break down exactly *why* a profile was flagged as a catfish, completely removing the "black box" of machine learning.
* **Heuristic Fallback Engine**: Blends raw ML probabilities with rule-based Z-score anomaly detection to ensure physically impossible behavior is always flagged.
* **Interactive Live Scanner**: A stunning, real-time local web UI that allows users to adjust behavioral sliders and watch the AI models vote on the threat level instantly.

## 🧠 The Machine Learning Pipeline (V32 ULTIMATE)
Our ultimate **V32** pipeline represents the pinnacle of our mathematical optimization:
1. **Dataset Generation**: 50,000 rows of synthetically modeled dating app behavior reflecting a real-world highly imbalanced distribution (93% Genuine, 7% Catfish).
2. **Advanced Feature Engineering**: 
   * `engagement_score`: Messages sent relative to time spent online.
   * `swipe_intensity`: Swipes per minute.
   * `bio_efficiency`: Bio character length relative to messaging activity.
3. **Data Balancing (SMOTE-Tomek)**: Utilizes Synthetic Minority Over-sampling Technique combined with Tomek Links to mathematically balance the heavily skewed dataset without introducing noise.
4. **Dynamic PCA Integration**: Principal Component Analysis perfectly retains 95% of mathematical variance to optimize the Neural Network's training time and accuracy.
5. **Hyperparameter Tuning**: Dynamically tuned using `RandomizedSearchCV` across a Stratified K-Fold cross-validation to maximize F1-Scores.
6. **Threshold Optimization**: Dynamically searches across a wide range of probability thresholds (0.10 to 0.90) to mathematically balance Precision and Recall.

## 📈 Performance & Evaluation
Because our problem is heavily imbalanced, we evaluate our AI primarily using **F1-Score**, which perfectly balances catching scammers (Recall) without falsely banning innocent users (Precision).
* **Top Performers**: Both XGBoost and Random Forest achieve ~99% F1-Scores, proving that they have essentially "solved" the mathematical behavioral differences between real and fake users.
* **Diagnostics Gallery**: The system evaluates models using comprehensive PR Curves, ROC-AUC Curves, and Probability Distribution histograms to prove mathematical separability.

## 💻 Running the Live Web Application

The project includes a fully functional, highly polished local web server with a modern glassmorphism UI. To ensure anyone can run this seamlessly, we have **pre-compiled all 6 Machine Learning models** (a 58MB bundle) and included them in the repository. **You can launch the website instantly with zero training time!**

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
*Note: Do not run `app.py` directly. The `run_web.py` script is our official launcher. It will dynamically load the AI brain, boot the local Flask server, wait for a successful connection, and then automatically pop open your web browser to `http://127.0.0.1:5000/`.*

## 📂 Repository Structure & Architecture
* `/WIA1006_Catfish_Group7_V32_Ultimate.ipynb` - The primary, fully-documented ML pipeline Colab notebook.
* `/dating_app_behavior_dataset.csv` - The core dataset containing 50,000 raw behavioral profiles.
* `/run_web.py` - Master Python script used to launch the interactive website safely.
* `/catfish_core.py` - The core backend engine handling the Machine Learning pipeline, feature engineering, and the heuristic scoring logic.
* `/webapp/app.py` - Flask server handling all backend routing, JSON endpoints, and API logic.
* `/webapp/templates/` - HTML files housing the frontend UI for the interactive web scanner, Explainability Center, and Model Battle Arena.
* `/webapp/static/` - Custom CSS stylesheets, SVGs, and dynamic frontend JavaScript.
* `/artifacts/` - Houses the pre-compiled `detector_bundle.pkl` (the trained AI models) and auto-generated matplotlib diagnostic plots.

---
<div align="center">
  <i>Developed with ❤️ for WIA1006 Machine Learning.</i>
</div>
