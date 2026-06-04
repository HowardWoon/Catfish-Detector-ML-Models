import json

def apply_class_injection():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        if "class GMMClassifier(BaseEstimator, ClassifierMixin):" in source and "class KMeansClassifier(BaseEstimator, ClassifierMixin):" in source:
            # We found the cell containing the classes! Let's replace the whole cell's content with our enhanced classes.
            # But wait, does this cell contain other things?
            # In the notebook, this is Cell 12: ML Pipeline Initialization. It contains ALL the imports, classes, and base_models.
            # It's safer to just replace the specific lines of the classes.
            
            gmm_code = """class GMMClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, n_components=2, covariance_type='diag', reg_covar=1e-3, random_state=42):
    self.n_components = n_components
    self.covariance_type = covariance_type
    self.reg_covar = reg_covar
    self.random_state = random_state

  def fit(self, X, y):
    import numpy as np
    from sklearn.mixture import GaussianMixture
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
      gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        random_state=self.random_state,
        max_iter=500,
        tol=1e-4,
        reg_covar=self.reg_covar,
        n_init=3,
      )
      gmm.fit(X_c)
      self.gmms_[c] = gmm
      self.log_priors_[c] = np.log(len(X_c) / len(y))
    return self

  def predict(self, X):
    import numpy as np
    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

  def predict_proba(self, X):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    log_joint = np.zeros((len(X), 2))
    for i, c in enumerate(self.classes_):
      log_joint[:, i] = self.gmms_[c].score_samples(X) + self.log_priors_[c]
    log_joint -= log_joint.max(axis=1, keepdims=True)
    probs = np.exp(log_joint)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs"""

            kmeans_code = """class KMeansClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, n_clusters=2, refine_iters=15, temperature=0.5, random_state=42):
    self.n_clusters = n_clusters
    self.refine_iters = refine_iters
    self.temperature = temperature
    self.random_state = random_state

  def _build_centroids(self, X, y):
    import numpy as np
    from sklearn.cluster import KMeans
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
    import numpy as np
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
    import numpy as np
    d = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
    return np.argmin(d, axis=1)

  def predict(self, X):
    import numpy as np
    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

  def predict_proba(self, X):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
    temp = max(float(self.temperature), 1e-6)
    neg_dist = -distances / temp
    exp_neg = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
    weights = exp_neg / exp_neg.sum(axis=1, keepdims=True)
    catfish_fracs = np.array([self.cluster_catfish_frac_[i] for i in range(self.n_clusters)])
    catfish_prob = np.clip(weights.dot(catfish_fracs), 0.0, 1.0)
    return np.column_stack([1.0 - catfish_prob, catfish_prob])"""

            import re
            
            # Use regex to replace the entire class definitions
            source = re.sub(r"class GMMClassifier\(BaseEstimator, ClassifierMixin\):.*?return np\.column_stack\(\[1\.0 \- catfish_prob, catfish_prob\]\)", gmm_code, source, flags=re.DOTALL)
            
            source = re.sub(r"class KMeansClassifier\(BaseEstimator, ClassifierMixin\):.*?return np\.column_stack\(\[1\.0 \- catfish_prob, catfish_prob\]\)", kmeans_code, source, flags=re.DOTALL)

            cell['source'] = [line + '\n' for line in source.split('\n')]
            
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Classes injected successfully!")

if __name__ == '__main__':
    apply_class_injection()
