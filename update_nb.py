import json
import sys

nb_path = 'WIA1006_Catfish_Group7_Ultimate.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Cell 3: Imports
        if 'CELL 3 | MASTER IMPORTS' in source:
            source = source.replace('from sklearn.naive_bayes     import GaussianNB\n',
                                    'from sklearn.mixture         import GaussianMixture\n')
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
            
        # Cell 12: Pipeline Initialization
        elif 'CELL 12 | ML Pipeline Initialization' in source:
            source = source.replace('from sklearn.naive_bayes    import GaussianNB\n',
                                    'from sklearn.mixture        import GaussianMixture\n')
            
            # We need to replace the models and params again, and insert GMMClassifier
            # Let's insert GMMClassifier right before KMeansClassifier
            gmm_class = '''
class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        
    def fit(self, X, y):
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.gmm.fit(X)
        self.cluster_mapping_ = {}
        clusters = self.gmm.predict(X)
        for i in range(self.n_components):
            mask = clusters == i
            if mask.sum() > 0:
                self.cluster_mapping_[i] = int(round(y[mask].mean()))
            else:
                self.cluster_mapping_[i] = 0
        return self
        
    def predict(self, X):
        clusters = self.gmm.predict(X)
        return np.array([self.cluster_mapping_[c] for c in clusters])
        
    def predict_proba(self, X):
        preds = self.predict(X)
        probas = np.zeros((len(X), 2))
        for i, p in enumerate(preds):
            probas[i, p] = 1.0
        return probas
'''
            source = source.replace('class KMeansClassifier(BaseEstimator, ClassifierMixin):',
                                    gmm_class + '\nclass KMeansClassifier(BaseEstimator, ClassifierMixin):')
            
            old_base_models = '''base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42),
    'KMeans + PCA': Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('kmeans', KMeansClassifier(random_state=42))]),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=500, random_state=42)
}'''
            new_base_models = '''base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=42),
    'Gaussian Mixture Model': GMMClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42),
    'KMeans + PCA': Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('kmeans', KMeansClassifier(random_state=42))]),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=500, random_state=42)
}'''
            source = source.replace(old_base_models, new_base_models)
            
            old_param_grids = '''param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 0.5, 1, 5], "penalty": ["l2"]},
    "Decision Tree": {"max_depth": [5, 10, 15, None], "min_samples_split": [5, 10, 20], "min_samples_leaf": [2, 4, 8]},
    "Gaussian Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "KMeans + PCA": {"kmeans__n_clusters": [2, 3, 4, 5]},
    "MLP Neural Network": {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.0001, 0.001]}
}'''
            new_param_grids = '''param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 0.5, 1, 5], "penalty": ["l2"]},
    "Decision Tree": {"max_depth": [5, 10, 15, None], "min_samples_split": [5, 10, 20], "min_samples_leaf": [2, 4, 8]},
    "Gaussian Mixture Model": {"n_components": [2, 3, 4]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "KMeans + PCA": {"kmeans__n_clusters": [2, 3, 4, 5]},
    "MLP Neural Network": {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.0001, 0.001]}
}'''
            source = source.replace(old_param_grids, new_param_grids)
            cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated notebook')
