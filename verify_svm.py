import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('dating_app_behavior_dataset.csv')
X = df.drop(columns=['Target'])
y = df['Target']

# Simplify preprocessing just to test SVM capacity
print("Quick preprocessing...")
# Keep numeric cols
X = X.select_dtypes(include=[np.number]).fillna(0)

print("Training SVM with MinMaxScaler + RBF Kernel...")
subset_idx = np.random.choice(len(X), size=15000, replace=False)
X_sub = X.iloc[subset_idx]
y_sub = y.iloc[subset_idx]

svm_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
])

param_grid = {
    'svm__C': [10, 50, 100],
    'svm__gamma': ['scale', 0.1, 0.5],
    'svm__kernel': ['rbf']
}

rs = RandomizedSearchCV(svm_pipeline, param_grid, n_iter=3, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
rs.fit(X_sub, y_sub)

print(f"Best Params: {rs.best_params_}")
print(f"Best CV F1-Macro Score: {rs.best_score_:.4f}")

# Evaluate on a holdout
holdout_idx = np.setdiff1d(np.arange(len(X)), subset_idx)
X_test = X.iloc[holdout_idx][:5000]
y_test = y.iloc[holdout_idx][:5000]

preds = rs.predict(X_test)
print("\nHoldout Performance:")
print(classification_report(y_test, preds))
