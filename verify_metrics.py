import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, accuracy_score, precision_score, recall_score
from catfish_core import load_dataset, engineer_features, prepare_features, GMMClassifier, KMeansClassifier

def eval_metrics(probs, y_te, name):
    pa, ra, ta = precision_recall_curve(y_te, probs)
    best_f1, best_t = -1, 0
    
    # 0.01 to 0.99 threshold constraint just like the app
    for p, r, t in zip(pa, ra, ta):
        if 0.01 <= t <= 0.99:
            if p+r > 0:
                fv = 2*p*r/(p+r)
                if fv > best_f1: 
                    best_f1, best_t = fv, t
                    
    preds = (probs >= best_t).astype(int)
    acc = accuracy_score(y_te, preds)
    rec = recall_score(y_te, preds)
    prec = precision_score(y_te, preds)
    roc = roc_auc_score(y_te, probs)
    print(f"\n--- {name} ---")
    print(f"Threshold: {best_t:.4f}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"F1-Score:  {best_f1*100:.2f}%")
    print(f"ROC-AUC:   {roc:.4f}")

print("Loading data...")
df = load_dataset()
df = engineer_features(df)
X, y = prepare_features(df)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_tr, y_tr)

print("Training GMM on SMOTE data...")
gmm = GMMClassifier(n_components=5, covariance_type='diag')
gmm.fit(X_sm[:15000], y_sm[:15000])  # Use subset for speed
eval_metrics(gmm.predict_proba(X_te)[:, 1], y_te, "Gaussian Mixture Model (SMOTE)")

print("Training KMeans (n_clusters=30) on SMOTE data...")
km = KMeansClassifier(n_clusters=30, temperature=0.1)
km.fit(X_sm[:15000], y_sm[:15000])
eval_metrics(km.predict_proba(X_te)[:, 1], y_te, "KMeans (SMOTE)")

print("Verification complete.")
