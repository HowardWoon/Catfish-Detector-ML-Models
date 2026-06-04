import json
import os

with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 12 (it's the code cell after the "Cell 12" markdown)
cell_12_index = -1
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '12' in ''.join(cell['source']) and 'Pipeline Initialization' in ''.join(cell['source']):
        cell_12_index = idx + 1
        break

perfect_cell_12 = """# ==============================================================================
# 🔥 CELL 12: ML PIPELINE INITIALIZATION
# ==============================================================================
# DESCRIPTION:
# This cell initializes the configurations for all 6 Machine Learning algorithms. 
# It sets up the hyperparameter grids, model wrappers, and tuning functions.
# ==============================================================================

import numpy as np
import warnings
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture        import GaussianMixture
from sklearn.svm            import SVC
from sklearn.cluster        import KMeans
from sklearn.pipeline       import Pipeline
from sklearn.decomposition  import PCA
from sklearn.base           import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

if 'X_train_bal' not in dir(): raise RuntimeError('❌ Run Cells 4–9 first.')

# Determine environment for optimal parallel jobs
IN_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in dir() else False
N_JOBS = -1 if IN_COLAB else 1
RANDOM_STATE = 42

class GMMClassifier(BaseEstimator, ClassifierMixin):
  \"\"\"Class-conditional GMM: separate mixture per class (supervised generative).\"\"\"
  def __init__(self, n_components=2, covariance_type='diag', reg_covar=1e-3, random_state=42):
    self.n_components = n_components
    self.covariance_type = covariance_type
    self.reg_covar = reg_covar
    self.random_state = random_state

  def fit(self, X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    self.classes_ = np.array([0, 1])
    self.gmms_ = {}
    self.log_priors_ = {}
    n_features = X.shape[1]
    cov_type = self.covariance_type
    if cov_type == 'full' and n_features > 40:
      cov_type = 'diag'
    for c in self.classes_:
      X_c = X[y == c]
      if len(X_c) == 0:
        raise ValueError(f'No training samples for class {c}')
      k = min(self.n_components, max(1, len(X_c) // 30))
      gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=self.random_state, max_iter=500, tol=1e-4, reg_covar=self.reg_covar, n_init=3)
      gmm.fit(X_c)
      self.gmms_[c] = gmm
      self.log_priors_[c] = np.log(len(X_c) / len(y))
    return self

  def predict(self, X):
    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

  def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float64)
    log_joint = np.zeros((len(X), 2))
    for i, c in enumerate(self.classes_):
      log_joint[:, i] = self.gmms_[c].score_samples(X) + self.log_priors_[c]
    log_joint -= log_joint.max(axis=1, keepdims=True)
    probs = np.exp(log_joint)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

class KMeansClassifier(BaseEstimator, ClassifierMixin):
  \"\"\"Label-aware KMeans with tunable refinement and softmax temperature.\"\"\"
  def __init__(self, n_clusters=2, refine_iters=15, temperature=0.5, random_state=42):
    self.n_clusters = n_clusters
    self.refine_iters = refine_iters
    self.temperature = temperature
    self.random_state = random_state

  def _build_centroids(self, X, y):
    centroids = [X[y == 0].mean(axis=0)]
    X_cat = X[y == 1]
    sub_k = max(1, self.n_clusters - 1)
    if len(X_cat) >= sub_k and sub_k > 1:
      km = KMeans(n_clusters=sub_k, random_state=self.random_state, n_init=15)
      km.fit(X_cat)
      centroids.extend(km.cluster_centers_)
    else:
      centroids.append(X_cat.mean(axis=0) if len(X_cat) else centroids[0])
    return np.vstack(centroids[: self.n_clusters])

  def fit(self, X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    self.classes_ = np.array([0, 1])
    self.centroids_ = self._build_centroids(X, y)
    self.n_clusters = len(self.centroids_)
    labels = self._nearest_centroid(X)
    min_pts = max(5, len(X) // (self.n_clusters * 50))
    for _ in range(self.refine_iters):
      for i in range(self.n_clusters):
        mask = labels == i
        if mask.sum() >= min_pts:
          self.centroids_[i] = X[mask].mean(axis=0)
      labels = self._nearest_centroid(X)
    self.cluster_catfish_frac_ = {}
    for i in range(self.n_clusters):
      mask = labels == i
      self.cluster_catfish_frac_[i] = float(y[mask].mean()) if mask.any() else 0.5
    return self

  def _nearest_centroid(self, X):
    d = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
    return np.argmin(d, axis=1)

  def predict(self, X):
    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

  def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float64)
    distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
    temp = max(float(self.temperature), 1e-6)
    neg_dist = -distances / temp
    exp_neg = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
    weights = exp_neg / exp_neg.sum(axis=1, keepdims=True)
    catfish_fracs = np.array([self.cluster_catfish_frac_[i] for i in range(self.n_clusters)])
    catfish_prob = np.clip(weights.dot(catfish_fracs), 0.0, 1.0)
    return np.column_stack([1.0 - catfish_prob, catfish_prob])

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=RANDOM_STATE),
    'Gaussian Mixture Model': GMMClassifier(random_state=RANDOM_STATE),
    'KMeans + PCA': Pipeline([('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)), ('kmeans', KMeansClassifier(random_state=RANDOM_STATE))]),
    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=200, batch_size=512, random_state=RANDOM_STATE),
    'Support Vector Machine': SVC(max_iter=3000, probability=True, class_weight='balanced', random_state=RANDOM_STATE),
    'KMeans': KMeansClassifier(random_state=RANDOM_STATE)
}

param_grids = {
    \"Logistic Regression\": {\"C\": [0.5, 1, 2], \"penalty\": [\"l2\"]},
    \"Decision Tree\": {\"max_depth\": [8, 12, 16], \"min_samples_split\": [5, 10]},
    \"Gaussian Mixture Model\": {\"n_components\": [2, 3], \"covariance_type\": [\"full\", \"tied\"]},
    \"KMeans + PCA\": {\"kmeans__n_clusters\": [2, 3]},
    \"MLP Neural Network\": {\"hidden_layer_sizes\": [(128, 64), (64, 32), (256, 128, 64)], \"alpha\": [0.0005, 0.001, 0.005]}
}

param_grids_orig = {
    \"Support Vector Machine\": {\"C\": [10], \"gamma\": [\"auto\"]},
    \"KMeans\": {\"n_clusters\": [30, 40, 50]}
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
models = {}

def tune_model(name):
    print(f'🔄 Tuning {name} on SMOTE data...')
    rs = RandomizedSearchCV(base_models[name], param_grids[name], n_iter=10, cv=cv, scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=N_JOBS)
    subset_size = min(15000, len(X_train_bal))
    subset_idx = np.random.choice(len(X_train_bal), size=subset_size, replace=False)
    x_fast = X_train_bal.iloc[subset_idx] if hasattr(X_train_bal, 'iloc') else X_train_bal[subset_idx]
    y_fast = y_train_bal.iloc[subset_idx] if hasattr(y_train_bal, 'iloc') else y_train_bal[subset_idx]
    rs.fit(x_fast, y_fast)
    models[name] = rs.best_estimator_
    print(f'✅ {name} best params: {rs.best_params_}')

def tune_model_orig(name):
    print(f'🔄 Tuning {name} on ORIGINAL data...')
    rs = RandomizedSearchCV(base_models[name], param_grids_orig[name], n_iter=10, cv=cv, scoring='f1_macro', random_state=RANDOM_STATE, n_jobs=N_JOBS)
    subset_size = min(15000, len(X_train_arr))
    subset_idx = np.random.choice(len(X_train_arr), size=subset_size, replace=False)
    x_fast = X_train_arr[subset_idx]
    y_fast = y_train_arr[subset_idx]
    rs.fit(x_fast, y_fast)
    models[name] = rs.best_estimator_
    print(f'✅ {name} best params: {rs.best_params_}')

print(f'✅ ML Pipeline Initialized. Ready to train on {X_train_bal.shape[1]} features.')
"""

if cell_12_index != -1:
    nb['cells'][cell_12_index]['source'] = [line + '\\n' for line in perfect_cell_12.split('\\n')]
    nb['cells'][cell_12_index]['source'][-1] = nb['cells'][cell_12_index]['source'][-1].strip()

# Now add CELL 25!
cell_25_markdown = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🌐 Cell 25 — Launch Live Web Application\n",
    "> **Link to Website:** Run this cell to host the interactive Flask web application directly from the notebook using ngrok! The website will use the exact models trained in this notebook."
   ]
}

cell_25_code = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 🌐 CELL 25: LAUNCH LIVE WEB APPLICATION\n",
    "# ==============================================================================\n",
    "# DESCRIPTION:\n",
    "# This cell bridges the Jupyter Notebook and the Flask Web Application.\n",
    "# It automatically installs pyngrok, starts the backend, and provides a public\n",
    "# URL so anyone can use the Live Catfish Scanner interface!\n",
    "# ==============================================================================\n",
    "\n",
    "import os, subprocess, time\n",
    "try:\n",
    "    from pyngrok import ngrok\n",
    "except ImportError:\n",
    "    import sys\n",
    "    !{sys.executable} -m pip install pyngrok flask flask-cors\n",
    "    from pyngrok import ngrok\n",
    "\n",
    "print('🔗 Creating ngrok tunnel...')\n",
    "try:\n",
    "    # Disconnect any existing tunnels\n",
    "    ngrok.kill()\n",
    "    public_url = ngrok.connect(5000).public_url\n",
    "    print(f'\\n✅ \\033[1mWEB APP IS LIVE AT:\\033[0m {public_url}')\n",
    "    print('🚀 Booting Flask server... (Click the link above to view)')\n",
    "    \n",
    "    # Write a quick script to run the app if it doesn't exist\n",
    "    if not os.path.exists('run_web.py'):\n",
    "        print('⚠️ run_web.py not found in current directory! Assuming we are in a bare environment.')\n",
    "    else:\n",
    "        subprocess.Popen(['python', 'run_web.py'])\n",
    "except Exception as e:\n",
    "    print('❌ Failed to start ngrok/flask:', e)\n"
   ]
}

# Update Cell 24 to also export detector_bundle.pkl
cell_24_idx = -1
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '24' in ''.join(cell['source']):
        cell_24_idx = idx + 1
        break

if cell_24_idx != -1:
    c24_source = ''.join(nb['cells'][cell_24_idx]['source'])
    if 'detector_bundle.pkl' not in c24_source:
        insert_idx = -1
        for i, line in enumerate(nb['cells'][cell_24_idx]['source']):
            if "print('   💾 Pipeline assets saved to', EXP)" in line:
                insert_idx = i
                break
        if insert_idx != -1:
            bundle_code = [
                "    # Build artifact bundle for website link\n",
                "    try:\n",
                "        from catfish_core import DetectorArtifacts\n",
                "        import copy\n",
                "        # Strip dataset to save space\n",
                "        safe_models = copy.copy(models)\n",
                "        arts = DetectorArtifacts(dataset_shape=df.shape, catfish_ratio=float((df['Target']==1).mean()), feature_names=FEATURE_NAMES, numerical_columns=NUM_COLS, population_stats=POP, scaler=scaler, pca=pca if 'pca' in globals() else None, models=safe_models, best_thresholds=BEST_THRESHOLDS, selected_features=SELECTED_FEATURES)\n",
                "        joblib.dump(arts, os.path.join(EXP, 'detector_bundle.pkl'))\n",
                "        print('   💾 detector_bundle.pkl (Website Link Ready)')\n",
                "    except Exception as e:\n",
                "        print(f'Warning: could not bundle for website: {e}')\n"
            ]
            nb['cells'][cell_24_idx]['source'] = nb['cells'][cell_24_idx]['source'][:insert_idx+1] + bundle_code + nb['cells'][cell_24_idx]['source'][insert_idx+1:]


nb['cells'].append(cell_25_markdown)
nb['cells'].append(cell_25_code)

with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('✅ Notebook updated successfully!')
