import json

def restore_and_enhance():
    # 1. Read the original notebook text to get Cell 12
    with open('notebook_text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cell12_start = -1
    cell12_end = -1
    for i, line in enumerate(lines):
        if 'CELL 12 (code)' in line:
            cell12_start = i + 1
        if 'CELL 12b (markdown)' in line:
            cell12_end = i - 1
            break
            
    cell12_code = "".join(lines[cell12_start:cell12_end])
    
    # 2. Modify the base_models block inside cell12_code
    
    old_block = """base_models = {
    'Logistic Regression': LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=RANDOM_STATE),
    'Gaussian Mixture Model': GMMClassifier(random_state=RANDOM_STATE),
    'KMeans + PCA': Pipeline([('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)), ('kmeans', KMeansClassifier(random_state=RANDOM_STATE))]),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=200, batch_size=512, random_state=RANDOM_STATE),
    'Support Vector Machine': SVC(max_iter=3000, probability=True, class_weight='balanced', random_state=RANDOM_STATE),
    'KMeans': KMeansClassifier(random_state=RANDOM_STATE)
}

param_grids = {
    "Logistic Regression": {"C": [0.5, 1, 2], "penalty": ["l1", "l2"]},
    "Decision Tree": {"max_depth": [8, 12, 16], "min_samples_split": [5, 10]},
    "Gaussian Mixture Model": {"n_components": [2, 3], "covariance_type": ["full", "tied"]},
    "MLP Neural Network": {"hidden_layer_sizes": [(128, 64), (64, 32)], "alpha": [0.001, 0.005]},
    "Support Vector Machine": {"C": [10], "gamma": ["auto"]},
    "KMeans": {"n_clusters": [30, 40, 50]}
}"""

    new_block = """base_models = {
    'Logistic Regression': LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=RANDOM_STATE),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=200, batch_size=512, random_state=RANDOM_STATE),
    'Gaussian Mixture Model': Pipeline([('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)), ('gmm', GMMClassifier(random_state=RANDOM_STATE))]),
    'KMeans': KMeansClassifier(random_state=RANDOM_STATE)
}

base_models_orig = {
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000, tol=1e-3, cache_size=2000)
}

param_grids = {
    "Logistic Regression": {"C": [0.5, 1, 2], "penalty": ["l2"]},
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
    models[name] = gs.best_estimator_"""

    cell12_code = cell12_code.replace(old_block, new_block)
    
    # Also add plot_prob_scatter_2d to the end of the cell
    plot_fn = """
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
    cell12_code += plot_fn

    # 3. Inject it back into the actual Jupyter Notebook
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if 'ML Pipeline Initialized.' in source:
                cell['source'] = [line + '\n' for line in cell12_code.split('\n')]
                break
                
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("SUCCESS")

if __name__ == '__main__':
    restore_and_enhance()
