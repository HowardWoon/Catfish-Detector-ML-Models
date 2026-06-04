import json
import re

def patch():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            
            # Cell 12: Base Models and Grids
            if 'base_models =' in source and 'ML Pipeline Initialized' in source:
                new_source = """
# Setup 6 ML Models
RANDOM_STATE = 42
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=RANDOM_STATE),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=200, random_state=RANDOM_STATE),
    'Gaussian Mixture Model': Pipeline([('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)), ('gmm', GMMClassifier(random_state=RANDOM_STATE))]),
    'KMeans': KMeansClassifier(random_state=RANDOM_STATE)
}

base_models_orig = {
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000, tol=1e-3, cache_size=2000)
}

param_grids = {
    "Logistic Regression": {"C": [0.5, 1, 2], "penalty": ["l1", "l2"]},
    "Decision Tree": {"max_depth": [8, 12, 16], "min_samples_split": [5, 10]},
    "MLP Neural Network": {"hidden_layer_sizes": [(128, 64), (64, 32)], "alpha": [0.001, 0.005]},
    "Gaussian Mixture Model": {"gmm__n_components": [2, 3, 4], "gmm__covariance_type": ["diag", "full"], "gmm__reg_covar": [1e-4, 1e-3]},
    "KMeans": {"n_clusters": [20, 30, 40], "refine_iters": [10, 15], "temperature": [0.1, 0.25, 0.5]}
}

param_grids_orig = {
    "Support Vector Machine": {"C": [1, 10, 50], "gamma": ["scale", "auto"]}
}

models = {}
def tune_model(name):
    print(f'\\n⭐ Tuning {name} on SMOTE data...')
    gs = GridSearchCV(base_models[name], param_grids[name], cv=3, scoring='f1', n_jobs=-1)
    gs.fit(X_train_bal, y_train_bal)
    print(f'✅ {name} best params: {gs.best_params_}')
    models[name] = gs.best_estimator_

def tune_model_orig(name):
    print(f'\\n⭐ Tuning {name} on ORIGINAL data...')
    gs = GridSearchCV(base_models_orig[name], param_grids_orig[name], cv=3, scoring='f1', n_jobs=-1)
    gs.fit(X_train_arr, y_train_arr)
    print(f'✅ {name} best params: {gs.best_params_}')
    models[name] = gs.best_estimator_

print('🚀 ML Pipeline Initialized.')

def plot_prob_scatter_2d(X, y, probs, title):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=probs, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Predicted Probability (Catfish)")
    plt.title(f"{title} - Probability Distribution (PCA Reduced)", fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()
"""
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
            
            # GMM Cell
            elif 'CELL 12c' in source or 'Tuning Gaussian Mixture Model' in source:
                new_source = source.replace("gmm.gmms_", "gmm.named_steps['gmm'].gmms_").replace("gmm.reg_covar", "gmm.named_steps['gmm'].reg_covar")
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                
            # KMeans Cell
            elif 'CELL 12e' in source or 'Tuning KMeans' in source:
                new_source = source.replace("tune_model_orig(name)", "tune_model(name)")
                cell['source'] = [line + '\n' for line in new_source.split('\n')]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched robustly!")

if __name__ == '__main__':
    patch()
