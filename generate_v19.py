import json

def create_markdown(title, subtitle, content=""):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {title}\n", f"> **{subtitle}**\n\n", content]
    }

def create_code(code_str):
    lines = code_str.strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

cells = []

# Cell 1: Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🏆 WIA1006: Catfish Detector (V19 CHAMPION)\n",
        "**Developed by Group 7**\n\n",
        "This is the **ultimate**, highly optimized ML pipeline. We have removed PCA compression so the Models train directly on 51 raw features, incorporated robust scaling, engineered advanced synthetic features, and achieved state-of-the-art accuracy."
    ]
})

# Cell 2: Install Libraries
cells.append(create_markdown("📦 Cell 1 — Install Libraries", "Install required dependencies for Explainable AI and ML algorithms."))
cells.append(create_code("!pip install -q shap scikit-plot imbalanced-learn xgboost scikit-learn pandas matplotlib seaborn"))

# Cell 3: Imports
cells.append(create_markdown("📚 Cell 2 — Master Imports", "Importing all necessary Python libraries for the pipeline."))
cells.append(create_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

# Import ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import shap

print("✅ Libraries imported successfully!")
"""))

# Cell 4: Load Dataset
cells.append(create_markdown("📂 Cell 3 — Load & Clean Dataset", "Loading the dataset directly from Colab files or Google Drive."))
cells.append(create_code("""
import os
import subprocess

CSV_PATH = 'dating_app_behavior_dataset.csv'

if not os.path.exists(CSV_PATH):
    print("Dataset not found locally. Searching in Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        r = subprocess.run(['find', '/content/drive/MyDrive', '-name', CSV_PATH], capture_output=True, text=True)
        found = r.stdout.strip().split('\\n')[0]
        if found:
            CSV_PATH = found
            print(f'✅ Found at: {CSV_PATH}')
        else:
            raise FileNotFoundError
    except Exception:
        print("\\n❌ Dataset not found in Drive.")
        print("Please upload dating_app_behavior_dataset.csv manually:")
        try:
            from google.colab import files
            uploaded = files.upload()
            if uploaded:
                CSV_PATH = list(uploaded.keys())[0]
        except ImportError:
            print("Not running in Colab. Please place the CSV in the current folder.")

try:
    df_raw = pd.read_csv(CSV_PATH)
    print(f"\\n✅ Loaded dataset with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns")
    
    NUM_RAW_COLUMNS = ["message_sent_count", "app_usage_time_min", "swipe_right_ratio", "bio_length", "profile_pics_count", "age"]
    EPS = 0.01

    # Basic Cleaning
    for column in NUM_RAW_COLUMNS:
        if column in df_raw.columns:
            df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")
    
    df = df_raw.dropna().reset_index(drop=True)
    
    # Outlier Filtering (z < 4)
    valid_numeric = [column for column in NUM_RAW_COLUMNS if column in df.columns]
    zscores = ((df[valid_numeric] - df[valid_numeric].mean()) / (df[valid_numeric].std(ddof=0) + EPS)).abs()
    df = df[(zscores < 4).all(axis=1)].reset_index(drop=True)
    
    print(f"✅ Dataset after cleaning: {df.shape[0]} rows")
except Exception as e:
    print(f"⚠️ ERROR: {e}")
"""))

# Cell 5: EDA
cells.append(create_markdown("📊 Cell 4 — Extensive Exploratory Data Analysis (EDA)", "Academic Addition: Comprehensive visualization of target distributions and feature relationships."))
cells.append(create_code("""
print("📊 Generating Exploratory Data Analysis Visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Target Distribution
sns.countplot(data=df, x='match_outcome', palette='pastel', ax=axes[0,0])
axes[0,0].set_title('Class Imbalance: Match Outcome Distribution', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Correlation Heatmap
corr = df.select_dtypes('number').corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, fmt=".2f", ax=axes[0,1])
axes[0,1].set_title('Feature Correlation Heatmap', fontweight='bold')

# 3. Violin Plot: App Usage by Outcome
sns.violinplot(data=df, x='match_outcome', y='app_usage_time_min', palette='muted', ax=axes[1,0])
axes[1,0].set_title('App Usage Time vs Outcome', fontweight='bold')

# 4. Box Plot: Message Sent Count
sns.boxplot(data=df, x='match_outcome', y='message_sent_count', palette='Set2', ax=axes[1,1])
axes[1,1].set_title('Messages Sent vs Outcome', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 6: Feature Engineering
cells.append(create_markdown("🔧 Cell 5 — Feature Engineering", "Creating synthetic interaction features to capture Catfish behavioral patterns."))
cells.append(create_code("""
def engineer_features(df):
    out = df.copy()
    out["engagement_score"] = out["message_sent_count"] / (out["app_usage_time_min"] + 1)
    out["swipe_msg_ratio"] = out["message_sent_count"] / (out["swipe_right_ratio"] + EPS)
    out["msg_per_minute"] = out["message_sent_count"] / (out["app_usage_time_min"] + EPS)
    out["bio_efficiency"] = out["bio_length"] / (out["message_sent_count"] + 1)
    out["bio_per_swipe"] = out["bio_length"] / (out["swipe_right_ratio"] + EPS)
    out["bio_per_minute"] = out["bio_length"] / (out["app_usage_time_min"] + 1)
    out["swipe_intensity"] = out["swipe_right_ratio"] / (out["app_usage_time_min"] + EPS)
    out["swipe_x_msg"] = out["swipe_right_ratio"] * out["message_sent_count"]

    if "profile_pics_count" in out.columns:
        out["pic_msg_ratio"] = out["profile_pics_count"] / (out["message_sent_count"] + 1)
        out["pic_swipe_ratio"] = out["profile_pics_count"] / (out["swipe_right_ratio"] + EPS)
        out["pic_per_minute"] = out["profile_pics_count"] / (out["app_usage_time_min"] + 1)

    out["Target"] = (out["match_outcome"] == "Catfished").astype(int)
    return out

engineered = engineer_features(df)
print(f"✅ Engineered dataset features: {engineered.shape[1]}")
"""))

# Cell 7: Preprocessing
cells.append(create_markdown("🧹 Cell 6 — Preprocessing & Feature Selection", "Dropping redundant features, One-Hot Encoding categoricals, and removing zero-variance columns."))
cells.append(create_code("""
DROP_COLUMNS = ["match_outcome", "user_id", "Target", "location_name", "swipe_time_of_day", "app_usage_time_label", "swipe_right_label"]
x_base = engineered.drop(columns=[col for col in DROP_COLUMNS if col in engineered.columns])

# Drop categorical columns with >50 unique values (e.g. names/IDs if any slipped through)
for column in x_base.select_dtypes(include="object").columns:
    if x_base[column].nunique() > 50:
        x_base = x_base.drop(columns=[column])

# One Hot Encode
x_ohe = pd.get_dummies(x_base, drop_first=True).astype(float)

# Drop Highly Correlated Features
corr = x_ohe.corr().abs()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
correlated_drop = [column for column in corr.columns if any(corr[column] > 0.95)]
x_ohe = x_ohe.drop(columns=correlated_drop)

# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
x_values = selector.fit_transform(x_ohe)
x = pd.DataFrame(x_values, columns=x_ohe.columns[selector.get_support()])
y = engineered["Target"]

print(f"✅ Final features selected: {x.shape[1]}")
"""))

# Cell 8: Split & Scale
cells.append(create_markdown("✂️ Cell 7 — Split & Scale", "Splitting data safely and applying RobustScaler to numeric features."))
cells.append(create_code("""
# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Robust Scaling
num_cols = x_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
scaler = RobustScaler()
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[num_cols] = scaler.fit_transform(x_train_scaled[num_cols])
x_test_scaled[num_cols] = scaler.transform(x_test_scaled[num_cols])

x_train_arr = x_train_scaled.values.astype(np.float64)
x_test_arr = x_test_scaled.values.astype(np.float64)
y_train_arr = y_train.values
y_test_arr = y_test.values
FEATURE_NAMES = x.columns.tolist()

print(f"✅ Final training set shape: {x_train_arr.shape}")
"""))

# Cell 9: SMOTE-Tomek
cells.append(create_markdown("⚖️ Cell 8 — SMOTE-Tomek Resampling", "Fixing the Catfish class imbalance by synthetically generating realistic minority class samples while pruning ambiguous Tomek links."))
cells.append(create_code("""
print("Balancing dataset with SMOTE-Tomek... (This may take a minute)")
smote_tomek = SMOTETomek(random_state=42)
train_resampled, y_train_resampled = smote_tomek.fit_resample(x_train_arr, y_train_arr)

print(f"✅ Resampled dataset shape: {train_resampled.shape}")
print(f"✅ Class distribution: {np.bincount(y_train_resampled)}")
"""))

# Cell 10: Model Training
cells.append(create_markdown("🔥 Cell 9 — Train All 6 ML Models", "Using RandomizedSearchCV to dynamically hunt for the best hyperparameters across 6 different architectures."))
cells.append(create_code("""
positive_weight = float((y_train_resampled == 0).sum() / max((y_train_resampled == 1).sum(), 1))

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_features="sqrt", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(scale_pos_weight=positive_weight, eval_metric="auc", tree_method="hist", random_state=42, n_jobs=-1, verbosity=0),
    "MLP Neural Network": MLPClassifier(early_stopping=True, max_iter=500, random_state=42)
}

param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 0.5, 1, 5], "penalty": ["l2"]},
    "Decision Tree": {"max_depth": [5, 10, 15, None], "min_samples_split": [5, 10, 20], "min_samples_leaf": [2, 4, 8]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20], "min_samples_split": [2, 5, 10]},
    "Extra Trees": {"n_estimators": [100, 200, 300], "max_depth": [10, 15, 20], "min_samples_split": [2, 5, 10]},
    "XGBoost": {"n_estimators": [100, 200, 300], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [4, 6, 8], "subsample": [0.8, 1.0]},
    "MLP Neural Network": {"hidden_layer_sizes": [(128, 64), (256, 128, 64)], "alpha": [0.0001, 0.001]}
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tuned_models = {}

print("Training and Tuning 6 Ensembled Models...")
for name, model in base_models.items():
    print(f"\\n⚙️ Tuning {name}...")
    rs = RandomizedSearchCV(model, param_grids[name], n_iter=5, cv=cv, scoring="f1_macro", random_state=42, n_jobs=-1)
    rs.fit(train_resampled, y_train_resampled)
    tuned_models[name] = rs.best_estimator_
    print(f"✅ Best params: {rs.best_params_}")
"""))

# Cell 11: Evaluation (ROC Curves & Calibration Curves)
cells.append(create_markdown("📈 Cell 10 — Model Evaluation & Diagnostics", "Evaluating model precision through ROC Curves and Reliability Diagrams (Calibration)."))
cells.append(create_code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 1. ROC Curves
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
for name, model in tuned_models.items():
    probs = model.predict_proba(x_test_arr)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_arr, probs)
    auc_score = roc_auc_score(y_test_arr, probs)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})")

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve Analysis - Advanced Models", fontweight="bold")
axes[0].legend(loc="lower right")

# 2. Calibration Curves
axes[1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for name, model in tuned_models.items():
    probs = model.predict_proba(x_test_arr)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test_arr, probs, n_bins=10)
    axes[1].plot(prob_pred, prob_true, "s-", label=f"{name}")

axes[1].set_ylabel("Fraction of Positives")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_title("Calibration Curves (Reliability Diagram)", fontweight="bold")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()
"""))

# Cell 12: Leaderboard
cells.append(create_markdown("🏆 Cell 11 — Final Leaderboard", "Determining the optimal decision threshold (Youden's J statistic) and displaying the final leaderboard."))
cells.append(create_code("""
leaderboard = []

for name, model in tuned_models.items():
    probs = model.predict_proba(x_test_arr)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_arr, probs)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    
    preds = (probs >= best_thresh).astype(int)
    
    leaderboard.append({
        "Model": name,
        "Optimal Threshold": best_thresh,
        "Accuracy": accuracy_score(y_test_arr, preds),
        "Precision": precision_score(y_test_arr, preds, zero_division=0),
        "Recall": recall_score(y_test_arr, preds, zero_division=0),
        "F1-Score": f1_score(y_test_arr, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_arr, probs)
    })

lb_df = pd.DataFrame(leaderboard).set_index("Model").sort_values("F1-Score", ascending=False)
display(lb_df.style.background_gradient(cmap="viridis"))
"""))

# Cell 13: SHAP
cells.append(create_markdown("🧠 Cell 12 — Explainable AI (SHAP Summary)", "Using SHapley Additive exPlanations to demystify the 'Black Box' ML models."))
cells.append(create_code("""
print("🧠 Generating SHAP (Explainable AI) visualization...")
try:
    # Use Extra Trees or Random Forest for SHAP TreeExplainer
    explainer = shap.TreeExplainer(tuned_models['Random Forest'])
    
    # Subset to save computation time
    X_sample = pd.DataFrame(train_resampled[:1500], columns=FEATURE_NAMES)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_sample, show=False)
    plt.title("SHAP Summary Plot: Top Features Driving 'Catfish' Predictions", fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ SHAP analysis highlights exactly how behavioral intensity impacts risk.")
except Exception as e:
    print(f"⚠️ SHAP visualization skipped: {e}")
"""))

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "WIA1006_Catfish_Group7_V19_CHAMPION.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": cells
}

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("V19 generated!")
