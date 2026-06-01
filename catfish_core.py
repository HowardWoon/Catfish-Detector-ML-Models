from __future__ import annotations
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


class ModelFunction:
    """Lightweight pickleable wrapper that exposes estimator prediction methods as
    callable functions. Keeps the original estimator but ensures the object used in
    artifacts behaves like a predictable function with `predict_proba` and `predict`.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)

class GMMClassifier(BaseEstimator, ClassifierMixin):
  """Class-conditional GMM: separate mixture per class (supervised generative)."""
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
  """Label-aware KMeans with tunable refinement and softmax temperature."""
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



BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dating_app_behavior_dataset.csv"
NOTEBOOK_PATH = BASE_DIR / "WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_BUNDLE_PATH = ARTIFACT_DIR / "detector_bundle.pkl"

RAW_INPUT_COLUMNS = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "bio_length",
    "message_sent_count",
]

NUM_RAW_COLUMNS = [
    "message_sent_count",
    "app_usage_time_min",
    "swipe_right_ratio",
    "bio_length",
    "profile_pics_count",
    "age",
]

DROP_COLUMNS = [
    "match_outcome",
    "user_id",
    "Target",
    "location_name",
    "swipe_time_of_day",
    "app_usage_time_label",
    "swipe_right_label",
]

EPS = 0.01


@dataclass
class DetectorArtifacts:
    dataset_shape: Tuple[int, int]
    class_counts: Dict[str, int]
    feature_names: List[str]
    num_cols: List[str]
    train_medians_raw: Dict[str, float]
    genuine_medians_raw: Dict[str, float]
    catfish_medians_raw: Dict[str, float]
    scaler: RobustScaler
    thresholds: Dict[str, float]
    models: Dict[str, Any]
    feature_importances: Dict[str, float]
    population_stats: Dict[str, Tuple[float, float]]
    notebook_cells: List[Dict[str, Any]]
    leaderboard: pd.DataFrame
    model_metrics: pd.DataFrame
    test_profiles: Dict[str, Dict[str, float]]


def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )


def load_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    df_raw = _safe_read_csv(csv_path)
    for column in NUM_RAW_COLUMNS:
        if column in df_raw.columns:
            df_raw[column] = pd.to_numeric(df_raw[column], errors="coerce")

    df = df_raw.dropna().reset_index(drop=True)
    valid_numeric = [column for column in NUM_RAW_COLUMNS if column in df.columns]
    if valid_numeric:
        zscores = ((df[valid_numeric] - df[valid_numeric].mean()) / (df[valid_numeric].std(ddof=0) + EPS)).abs()
        df = df[(zscores < 4).all(axis=1)].reset_index(drop=True)

    if "match_outcome" not in df.columns:
        raise RuntimeError("Expected a match_outcome column in the dataset.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop = [column for column in DROP_COLUMNS if column in df.columns]
    x_base = df.drop(columns=drop)

    for column in x_base.select_dtypes(include="object").columns.tolist():
        if x_base[column].nunique() > 50:
            x_base = x_base.drop(columns=[column])

    x_ohe = pd.get_dummies(x_base, drop_first=True).astype(float)

    if x_ohe.shape[1] > 1:
        corr = x_ohe.corr().abs()
        corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        correlated_drop = [column for column in corr.columns if any(corr[column] > 0.95)]
        if correlated_drop:
            x_ohe = x_ohe.drop(columns=correlated_drop)

    selector = VarianceThreshold(threshold=0.01)
    x_values = selector.fit_transform(x_ohe)
    kept_columns = x_ohe.columns[selector.get_support()].tolist()
    x = pd.DataFrame(x_values, columns=kept_columns)
    y = df["Target"].reset_index(drop=True)
    return x, y


def build_scanner_input(
    raw_values: Dict[str, float],
    train_medians_raw: Dict[str, float],
    feature_names: List[str],
    num_cols: List[str],
    scaler: RobustScaler,
) -> np.ndarray:
    row = dict(train_medians_raw)
    row.update(raw_values)

    a = float(row.get("app_usage_time_min", 0.0))
    s = float(row.get("swipe_right_ratio", 0.0))
    b = float(row.get("bio_length", 0.0))
    m = float(row.get("message_sent_count", 0.0))

    row["engagement_score"] = m / (a + 1)
    row["swipe_msg_ratio"] = m / (s + EPS)
    row["msg_per_minute"] = m / (a + EPS)
    row["bio_efficiency"] = b / (m + 1)
    row["bio_per_swipe"] = b / (s + EPS)
    row["bio_per_minute"] = b / (a + 1)
    row["swipe_intensity"] = s / (a + EPS)
    row["swipe_x_msg"] = s * m
    if "pic_msg_ratio" in feature_names:
        pics = float(row.get("profile_pics_count", 3.0))
        row["pic_msg_ratio"] = pics / (m + 1)
        row["pic_swipe_ratio"] = pics / (s + EPS)
        row["pic_per_minute"] = pics / (a + 1)

    input_frame = pd.DataFrame([{feature: row.get(feature, 0.0) for feature in feature_names}], columns=feature_names)
    input_frame[num_cols] = scaler.transform(input_frame[num_cols])
    
    # Clip extreme anomalies so ML models evaluate them at the 99th percentile boundary
    # rather than extrapolating wildly into uncharted numerical territory
    input_values = input_frame.values.astype(np.float64)
    np.clip(input_values, -5.0, 5.0, out=input_values)
    
    return input_values


def train_models(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    
    base_models = {
        # LR: lbfgs with max_iter=3000 converges reliably on this scaled dataset.
        "Logistic Regression": LogisticRegression(max_iter=3000, solver='lbfgs', class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_features="sqrt", random_state=42),
        "Gaussian Mixture Model": GMMClassifier(random_state=42),
        # SVM: use C=0.3 (strong regularization) to prevent Platt scaling overconfidence.
        # With only ~8000 training samples in 51-feature space, a linear-margin SVM
        # still provides useful signal without saturating to 0%/100% probabilities.
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42, max_iter=5000, C=0.3),
        "KMeans": KMeansClassifier(random_state=42),
        "MLP Neural Network": MLPClassifier(early_stopping=True, max_iter=200, random_state=42)
    }

    param_grids = {
        "Logistic Regression": {"C": [0.5, 1, 2]},
        "Decision Tree": {"max_depth": [8, 12]},
        "Gaussian Mixture Model": {"n_components": [2, 3, 4], "covariance_type": ["diag", "full"], "reg_covar": [1e-4, 1e-3]},
        "Support Vector Machine": {"C": [0.1, 0.3]},
        "KMeans": {"n_clusters": [2, 3, 4], "refine_iters": [10, 15, 20], "temperature": [0.25, 0.5, 1.0]},
        "MLP Neural Network": {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.001, 0.01]}
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tuned_models = {}

    n_iter_map = {
        "Logistic Regression": 8,
        "Decision Tree": 8,
        "Gaussian Mixture Model": 8,
        "Support Vector Machine": 6,
        "KMeans": 8,
        "MLP Neural Network": 8,
    }
    sample_map = {
        "Support Vector Machine": 8000,
        "Gaussian Mixture Model": 12000,
        "KMeans": 12000,
    }

    for name, model in base_models.items():
        print(f"Tuning {name}...")
        rs = RandomizedSearchCV(
            model,
            param_grids[name],
            n_iter=n_iter_map.get(name, 8),
            cv=cv,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )

        max_samples = sample_map.get(name, 15000)
        subset_size = min(max_samples, len(x_train))
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(x_train), size=subset_size, replace=False)
        x_fast = x_train.iloc[subset_idx] if isinstance(x_train, pd.DataFrame) else x_train[subset_idx]
        y_fast = y_train.iloc[subset_idx] if isinstance(y_train, pd.Series) else y_train[subset_idx]

        rs.fit(x_fast, y_fast)
        best = rs.best_estimator_
        best.fit(x_train, y_train)
        tuned_models[name] = best
        print(f"  Best params: {rs.best_params_}")

    return tuned_models


def find_thresholds(models: Dict[str, Any], x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        precision, recall, threshold_values = precision_recall_curve(y_test, probabilities)
        best_threshold = 0.65
        best_f1 = -1.0
        for index, threshold in enumerate(threshold_values):
            if 0.25 <= threshold <= 0.85:
                score_total = precision[index] + recall[index]
                f1_value = (2 * precision[index] * recall[index] / score_total) if score_total > 0 else 0.0
                if f1_value > best_f1:
                    best_f1 = f1_value
                    best_threshold = float(threshold)
        thresholds[name] = best_threshold
    return thresholds


def build_population_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    columns = [
        "message_sent_count",
        "app_usage_time_min",
        "swipe_right_ratio",
        "bio_length",
        "profile_pics_count",
        "likes_received",
        "mutual_matches",
    ]
    stats: Dict[str, Tuple[float, float]] = {}
    for column in columns:
        if column in df.columns:
            stats[column] = (float(df[column].mean()), float(df[column].std(ddof=0)))
    return stats


def behavioral_risk(
    raw_input: Dict[str, float],
    population_stats: Dict[str, Tuple[float, float]],
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compute a behavioral risk score (0–100) from raw slider input values.

    Uses direct z-score multipliers (same formula as the Colab notebook scanner)
    instead of exponential dampening. This ensures meaningful scores:
      - Genuine profiles:  5–20%
      - Borderline:       20–40%
      - Catfish signals:  40–80%+

    Returns:
        (risk_score, top_flags)  where top_flags is a list of (label, points) tuples.
    """
    a     = float(raw_input.get("app_usage_time_min", 0.0))
    s     = float(raw_input.get("swipe_right_ratio",  0.0))
    b     = float(raw_input.get("bio_length",         0.0))
    m     = float(raw_input.get("message_sent_count", 0.0))
    pics  = float(raw_input.get("profile_pics_count", 0.0))
    likes = float(raw_input.get("likes_received",     0.0))
    matches = float(raw_input.get("mutual_matches",   0.0))

    def zs(column: str, value: float) -> float:
        mean, std = population_stats.get(column, (0.0, 1.0))
        return (value - mean) / (std + EPS)

    # ── Compute individual z-score components ──────────────────────────
    z_msg         = max(0.0,  zs("message_sent_count", m))      # high msgs → suspicious
    z_swipe       = abs(zs("swipe_right_ratio", s))              # extreme swipe (either direction)
    z_bio_short   = max(0.0, -zs("bio_length", b))              # very short bio → suspicious
    z_bio_long    = max(0.0,  zs("bio_length", b))              # very long bio → mild flag
    z_app         = max(0.0,  zs("app_usage_time_min", a))      # excessive usage → mild flag
    z_pics        = max(0.0, -zs("profile_pics_count", pics))   # very few pics → suspicious
    z_matches_low = max(0.0, -zs("mutual_matches", matches))    # very few matches → suspicious

    # Engagement density: messages per minute relative to population
    eng          = m / (a + 1)
    msg_mu       = population_stats.get("message_sent_count", (50.0, 1.0))[0]
    app_mu       = population_stats.get("app_usage_time_min", (150.0, 1.0))[0]
    eng_pop      = msg_mu / (app_mu + 1)
    z_eng        = max(0.0, (eng - eng_pop) / (eng_pop + EPS))  # high density → suspicious

    # Likes-to-matches ratio: many likes but very few mutual matches → bot pattern
    likes_mu     = population_stats.get("likes_received",  (100.0, 1.0))[0]
    matches_mu   = population_stats.get("mutual_matches",  (14.0,  1.0))[0]
    lm_ratio     = likes / (matches + 1)
    lm_pop       = likes_mu / (matches_mu + 1)
    z_lm         = max(0.0, (lm_ratio - lm_pop) / (lm_pop + EPS))

    # ── Weighted risk components (weights reflect behavioral psychology impact) ──
    # Scale: each z-score unit × weight × 1 point → total normalized over 42 pts
    risk_components: Dict[str, float] = {
        "High message density"       : z_eng        * 6.5,
        "High message count"         : z_msg        * 5.0,
        "Suspiciously short bio"     : z_bio_short  * 4.5,
        "High likes vs low matches"  : z_lm         * 4.0,
        "Extreme swipe pattern"      : z_swipe      * 3.5,
        "Very few mutual matches"    : z_matches_low * 3.0,
        "Very few profile pics"      : z_pics        * 3.0,
        "Excessive app usage"        : z_app         * 2.0,
        "Overlong bio"               : z_bio_long    * 1.0,
    }

    raw_risk = sum(risk_components.values())
    # Normalize: divide by 42 so that a profile with ALL signals at z=2 hits ~100%
    risk = round(min(100.0, max(0.0, (raw_risk / 42.0) * 100.0)), 1)

    # Sort top flags that actually contributed meaningfully (> 0.5 pts)
    top_flags = sorted(((name, value) for name, value in risk_components.items() if value > 0.5), key=lambda pair: -pair[1])[:4]
    return risk, top_flags


def scan_input(
    raw_input: Dict[str, float],
    artifacts: DetectorArtifacts,
) -> Dict[str, Any]:
    vector = build_scanner_input(
        raw_input,
        artifacts.train_medians_raw,
        artifacts.feature_names,
        artifacts.num_cols,
        artifacts.scaler,
    )

    # 1. Get real Machine Learning probabilities from all 6 models
    thresholds = _normalize_legacy_model_names(dict(artifacts.thresholds))
    model_probs = {
        display_model_name(name): float(model.predict_proba(vector)[0][1])
        for name, model in artifacts.models.items()
    }

    # 2. Count how many models triggered their specific thresholds
    ml_votes = sum(1 for name, prob in model_probs.items() if prob >= thresholds.get(name, 0.40))
    
    # 3. Calculate average ML probability
    avg_ml_prob = sum(model_probs.values()) / len(model_probs) if model_probs else 0.0
    ml_score = avg_ml_prob * 100.0
    
    # 4. Get behavioral heuristic risk score for explainability
    heuristic_score, top_flags = behavioral_risk(raw_input, artifacts.population_stats)
    
    # 5. Blend behavioral heuristics with ML ensemble consensus.
    ml_weight = 0.35
    spread = float(np.std(list(model_probs.values()))) if model_probs else 0.0
    if spread >= 0.08:
        ml_weight = min(0.50, 0.35 + spread)
    blended_score = (heuristic_score * (1.0 - ml_weight)) + (ml_score * ml_weight)
    
    # If either signal is very strong, don't suppress it
    if heuristic_score >= 70.0 or ml_score >= 75.0:
        blended_score = max(blended_score, heuristic_score, ml_score * 0.7)

    behavioral_score = round(min(100.0, max(0.0, blended_score)), 1)
    
    # 6. Final verdict: catfish if behavioral score >= 30% OR majority ML vote
    if behavioral_score >= 30.0 or ml_votes > (len(model_probs) * 0.5):
        final_verdict = "CATFISH"
    else:
        final_verdict = "GENUINE"

    return {
        "vector": vector,
        "model_probs": model_probs,
        "ml_votes": ml_votes,
        "behavioral_score": behavioral_score,
        "top_flags": top_flags,
        "final_verdict": final_verdict,
    }


def _read_notebook_cells(notebook_path: Path = NOTEBOOK_PATH) -> List[Dict[str, Any]]:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    cells: List[Dict[str, Any]] = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)
        # Produce a cleaned, formal title that strips markdown headers,
        # leading emojis and punctuation so the UI shows a concise cell name.
        def _sanitize_title(s: str) -> str:
            for line in s.splitlines():
                candidate = line.strip()
                if not candidate:
                    continue
                # Remove markdown header markers (#), leading non-alphanum characters (emojis, bullets), and excess separators
                cleaned = candidate.lstrip('#').strip()
                # Remove leading punctuation/emojis
                cleaned = cleaned.lstrip(' -–—•*\u200b')
                # Collapse multiple spaces
                cleaned = ' '.join(cleaned.split())
                # Truncate long titles
                if len(cleaned) > 120:
                    cleaned = cleaned[:117].rstrip() + '...'
                return cleaned

        title = _sanitize_title(text) or f"Cell {index}"
        cells.append({"index": index, "type": cell_type, "title": title, "source": text})
    return cells


def _prepare_training_table(models: Dict[str, Any], thresholds: Dict[str, float], x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        threshold = thresholds[name]
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "Model": name,
                "Threshold": round(threshold, 4),
                "Accuracy": accuracy_score(y_test, predictions),
                "Recall": recall_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "F1-Score": f1_score(y_test, predictions),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
        )
    return pd.DataFrame(rows).set_index("Model").sort_values("F1-Score", ascending=False)


def _train_test_artifacts(df: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    RobustScaler,
    Dict[str, Any],
    Dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Tuple[float, float]],
    Dict[str, float],
    List[Dict[str, Any]],
    Dict[str, Dict[str, float]],
]:
    engineered = engineer_features(df)
    x, y = prepare_features(engineered)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_names = x.columns.tolist()
    num_cols = x_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    train_medians_raw = x_train.median().to_dict()

    x_train_labeled = x_train.copy()
    x_train_labeled["__label__"] = y_train.values
    genuine_medians_raw = x_train_labeled[x_train_labeled["__label__"] == 0].drop(columns=["__label__"]).median().to_dict()
    catfish_medians_raw = x_train_labeled[x_train_labeled["__label__"] == 1].drop(columns=["__label__"]).median().to_dict()

    scaler = RobustScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    x_train_scaled[num_cols] = scaler.fit_transform(x_train_scaled[num_cols])
    x_test_scaled[num_cols] = scaler.transform(x_test_scaled[num_cols])

    x_train_arr = x_train_scaled.values.astype(np.float64)
    x_test_arr = x_test_scaled.values.astype(np.float64)
    y_train_arr = y_train.values
    y_test_arr = y_test.values

    # Compute Feature Importances on pre-PCA data so Web App displays real feature names
    print("Computing feature importances on raw data...")
    train_resampled_raw, y_train_resampled_raw = SMOTETomek(random_state=42).fit_resample(x_train_arr, y_train_arr)
    importance_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )
    importance_model.fit(train_resampled_raw, y_train_resampled_raw)
    feature_importances = dict(zip(feature_names, importance_model.feature_importances_))

    print("Balancing data with SMOTE-Tomek...")
    train_resampled, y_train_resampled = SMOTETomek(random_state=42).fit_resample(x_train_arr, y_train_arr)

    print("Training and tuning models...")
    models = train_models(train_resampled, y_train_resampled)
    # Wrap estimators in a lightweight callable wrapper so the saved artifacts
    # expose consistent `predict_proba` and `predict` methods and remain pickleable.
    models = {name: ModelFunction(m) for name, m in models.items()}
    thresholds = find_thresholds(models, x_test_arr, y_test_arr)

    leaderboard = _prepare_training_table(models, thresholds, x_test_arr, y_test_arr)
    model_metrics = leaderboard.reset_index().copy()

    population_stats = build_population_stats(df)
    test_profiles = {
        "genuine_median": {
            "app_usage_time_min": float(genuine_medians_raw.get("app_usage_time_min", 0.0)),
            "swipe_right_ratio": float(genuine_medians_raw.get("swipe_right_ratio", 0.0)),
            "bio_length": float(genuine_medians_raw.get("bio_length", 0.0)),
            "message_sent_count": float(genuine_medians_raw.get("message_sent_count", 0.0)),
            "profile_pics_count": float(genuine_medians_raw.get("profile_pics_count", 0.0)),
            "likes_received": float(genuine_medians_raw.get("likes_received", 0.0)),
            "mutual_matches": float(genuine_medians_raw.get("mutual_matches", 0.0)),
        },
        "catfish_median": {
            "app_usage_time_min": float(catfish_medians_raw.get("app_usage_time_min", 0.0)),
            "swipe_right_ratio": float(catfish_medians_raw.get("swipe_right_ratio", 0.0)),
            "bio_length": float(catfish_medians_raw.get("bio_length", 0.0)),
            "message_sent_count": float(catfish_medians_raw.get("message_sent_count", 0.0)),
            "profile_pics_count": float(catfish_medians_raw.get("profile_pics_count", 0.0)),
            "likes_received": float(catfish_medians_raw.get("likes_received", 0.0)),
            "mutual_matches": float(catfish_medians_raw.get("mutual_matches", 0.0)),
        },
        "low_activity": {
            "app_usage_time_min": 30.0,
            "swipe_right_ratio": 0.10,
            "bio_length": 420.0,
            "message_sent_count": 8.0,
            "profile_pics_count": 1.0,
            "likes_received": 20.0,
            "mutual_matches": 2.0,
        },
        "high_activity": {
            "app_usage_time_min": 290.0,
            "swipe_right_ratio": 0.95,
            "bio_length": 20.0,
            "message_sent_count": 98.0,
            "profile_pics_count": 0.0,
            "likes_received": 190.0,
            "mutual_matches": 1.0,
        },
    }

    notebook_cells = _read_notebook_cells()

    # Generate diagnostic plots for the trained models and dataset, store in artifacts/plots
    try:
        plot_dir = ARTIFACT_DIR / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        sns.set(style="whitegrid")

        # Feature importances plot
        fi_items = sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:40]
        if fi_items:
            names, vals = zip(*fi_items)
            plt.figure(figsize=(10, max(4, len(names) * 0.25)))
            sns.barplot(x=list(vals), y=list(names), hue=list(names), palette="viridis", legend=False)
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.savefig(plot_dir / "feature_importances.png", dpi=150)
            plt.close()

        # ROC curves for each model
        for name, model in models.items():
            try:
                probs = model.predict_proba(x_test_arr)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_arr, probs)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test_arr, probs):.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve — {name}")
                plt.legend(loc="lower right")
                plt.tight_layout()
                safe_name = name.replace(" ", "_").replace('/', '_')
                plt.savefig(plot_dir / f"roc_{safe_name}.png", dpi=150)
                plt.close()
            except Exception:
                continue

        # Model probability distributions
        for name, model in models.items():
            try:
                probs = model.predict_proba(x_test_arr)[:, 1]
                plt.figure(figsize=(8, 4))
                sns.histplot(probs[y_test_arr == 0], color="C0", label="genuine", stat="density", kde=True, binwidth=0.02)
                sns.histplot(probs[y_test_arr == 1], color="C1", label="catfish", stat="density", kde=True, binwidth=0.02)
                plt.title(f"Model Probability Distribution — {name}")
                plt.legend()
                plt.tight_layout()
                safe_name = name.replace(" ", "_").replace('/', '_')
                plt.savefig(plot_dir / f"probs_{safe_name}.png", dpi=150)
                plt.close()
            except Exception:
                continue

        # Leaderboard bar chart (F1-Score)
        try:
            lb = model_metrics.copy()
            plt.figure(figsize=(8, max(3, len(lb) * 0.5)))
            sns.barplot(x="F1-Score", y="Model", hue="Model", data=lb.sort_values("F1-Score", ascending=False), palette="magma", legend=False)
            plt.title("Model Leaderboard — F1-Score")
            plt.tight_layout()
            plt.savefig(plot_dir / "leaderboard_f1.png", dpi=150)
            plt.close()
        except Exception:
            pass
    except Exception:
        pass
    return (
        engineered,
        x_train_arr,
        x_test_arr,
        y_train_arr,
        y_test_arr,
        train_resampled,
        y_train_resampled,
        feature_names,
        num_cols,
        train_medians_raw,
        genuine_medians_raw,
        catfish_medians_raw,
        scaler,
        models,
        thresholds,
        leaderboard,
        model_metrics,
        population_stats,
        feature_importances,
        notebook_cells,
        test_profiles,
    )


def _normalize_legacy_model_names(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Rename retired bundle keys (e.g. 'KMeans + PCA' → 'KMeans')."""
    if "KMeans + PCA" in mapping and "KMeans" not in mapping:
        mapping["KMeans"] = mapping.pop("KMeans + PCA")
    return mapping


def _bundle_needs_retrain(artifacts: DetectorArtifacts) -> bool:
    """Retrain if cached bundle predates enhanced GMM/KMeans classifiers."""
    gmm = artifacts.models.get("Gaussian Mixture Model")
    if gmm is not None and not hasattr(gmm, "gmms_"):
        return True
    kmeans = artifacts.models.get("KMeans")
    if kmeans is not None and not hasattr(kmeans, "refine_iters"):
        return True
    return False


def load_artifacts() -> DetectorArtifacts:
    if ARTIFACT_BUNDLE_PATH.exists():
        artifacts = joblib.load(ARTIFACT_BUNDLE_PATH)
        artifacts.models = _normalize_legacy_model_names(dict(artifacts.models))
        artifacts.thresholds = _normalize_legacy_model_names(dict(artifacts.thresholds))
        if _bundle_needs_retrain(artifacts):
            print("⚠️  Stale model bundle — retraining all 6 models with latest pipeline...")
            ARTIFACT_BUNDLE_PATH.unlink(missing_ok=True)
        else:
            return artifacts

    df = load_dataset()
    (
        engineered,
        x_train_arr,
        x_test_arr,
        y_train_arr,
        y_test_arr,
        train_resampled,
        y_train_resampled,
        feature_names,
        num_cols,
        train_medians_raw,
        genuine_medians_raw,
        catfish_medians_raw,
        scaler,
        models,
        thresholds,
        leaderboard,
        model_metrics,
        population_stats,
        feature_importances,
        notebook_cells,
        test_profiles,
    ) = _train_test_artifacts(df)

    class_counts = {
        "catfished": int((df["match_outcome"] == "Catfished").sum()),
        "genuine": int((df["match_outcome"] != "Catfished").sum()),
    }
    dataset_shape = tuple(df.shape)

    artifacts = DetectorArtifacts(
        dataset_shape=dataset_shape,
        class_counts=class_counts,
        feature_names=feature_names,
        num_cols=num_cols,
        train_medians_raw={key: float(value) for key, value in train_medians_raw.items()},
        genuine_medians_raw={key: float(value) for key, value in genuine_medians_raw.items()},
        catfish_medians_raw={key: float(value) for key, value in catfish_medians_raw.items()},
        scaler=scaler,
        thresholds=thresholds,
        models=models,
        feature_importances=feature_importances,
        population_stats=population_stats,
        notebook_cells=notebook_cells,
        leaderboard=leaderboard,
        model_metrics=model_metrics,
        test_profiles=test_profiles,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, ARTIFACT_BUNDLE_PATH)
    return artifacts


def display_model_name(name: str) -> str:
    return "KMeans" if name == "KMeans + PCA" else name


def normalize_probability_map(values: Dict[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for name, value in values.items():
        normalized[display_model_name(name)] = float(value)
    return normalized


def build_model_details(
    model_probs: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, probability in model_probs.items():
        label = display_model_name(name)
        threshold = float(
            thresholds.get(name, thresholds.get(label, thresholds.get("KMeans + PCA", 0.40)))
        )
        flagged = probability >= threshold
        rows.append(
            {
                "name": label,
                "threshold": round(threshold, 4),
                "probability": round(float(probability), 4),
                "probability_pct": round(float(probability) * 100, 1),
                "verdict": "CATFISH" if flagged else "GENUINE",
                "flagged": flagged,
            }
        )
    return rows


def build_scan_html(
    *,
    behavioral_score: float,
    is_catfish: bool,
    ml_votes: int,
    model_count: int,
    raw_input: Dict[str, float],
    top_flags: List[Tuple[str, float]],
    model_details: List[Dict[str, Any]],
) -> str:
    title = "HIGH RISK: CATFISH DETECTED" if is_catfish else "LOW RISK: LIKELY GENUINE"
    banner = "#b71c1c" if is_catfish else "#1b5e20"
    bar_color = "#ef5350" if behavioral_score > 60 else ("#f59e0b" if behavioral_score > 30 else "#22c55e")

    flags_li = "".join(
        f'<li style="margin:4px 0;font-size:13px;color:#1f2937;"><span style="margin-right:6px;">⚠️</span>'
        f'<b>{label}</b>: {score:.2f} pts</li>'
        for label, score in top_flags
    ) or '<li style="color:#1f2937;font-size:13px;">✅ No significant behavioral red flags detected.</li>'

    ml_rows = "".join(
        f'<tr style="background:{"#fee2e2" if row["flagged"] else "#ecfdf5"};">'
        f'<td style="padding:10px 12px;font-size:13px;font-weight:700;color:#111827;">{row["name"]}</td>'
        f'<td style="padding:10px 12px;font-size:13px;text-align:center;font-weight:600;color:#374151;">{row["threshold"]:.3f}</td>'
        f'<td style="padding:10px 12px;font-size:13px;text-align:center;font-weight:700;color:{"#b91c1c" if row["flagged"] else "#15803d"};">'
        f'{row["probability_pct"]:.1f}%</td>'
        f'<td style="padding:10px 12px;font-size:13px;text-align:center;font-weight:700;color:{"#b91c1c" if row["flagged"] else "#15803d"};">'
        f'{"🚨 CATFISH" if row["flagged"] else "✅ GENUINE"}</td></tr>'
        for row in model_details
    )

    inp = raw_input
    return f"""
<div class="catfish-scan-report" style="font-family:Inter,Arial,sans-serif;max-width:760px;margin:16px auto;border-radius:14px;overflow:hidden;box-shadow:0 8px 28px rgba(0,0,0,0.18);border:1px solid #e5e7eb;">
  <div style="background:{banner};padding:22px;text-align:center;">
    <h2 style="color:#ffffff;margin:0;font-size:22px;font-weight:800;letter-spacing:0.5px;">{title}</h2>
    <div style="font-size:52px;font-weight:800;color:#ffffff;margin:12px 0;">{behavioral_score:.1f}%</div>
    <div style="background:rgba(255,255,255,0.35);border-radius:8px;height:14px;width:82%;margin:0 auto;">
      <div style="background:{bar_color};width:{int(behavioral_score)}%;height:14px;border-radius:8px;"></div>
    </div>
    <div style="color:#ffffff;font-size:12px;margin-top:8px;font-weight:600;">Blended Threat Score | Decision Threshold: 30%</div>
  </div>
  <div style="background:#ffffff;padding:18px;border-bottom:1px solid #e5e7eb;">
    <div style="font-size:14px;font-weight:800;color:#0f172a;margin-bottom:8px;">📊 Input Values</div>
    <div style="font-size:13px;color:#1f2937;line-height:1.6;">
      App Usage: <b>{inp.get("app_usage_time_min", 0):.0f} min</b> |
      Swipe Ratio: <b>{inp.get("swipe_right_ratio", 0):.2f}</b> |
      Bio: <b>{inp.get("bio_length", 0):.0f} chars</b><br>
      Messages: <b>{inp.get("message_sent_count", 0):.0f}</b> |
      Photos: <b>{inp.get("profile_pics_count", 0):.0f}</b> |
      Likes: <b>{inp.get("likes_received", 0):.0f}</b> |
      Matches: <b>{inp.get("mutual_matches", 0):.0f}</b>
    </div>
  </div>
  <div style="background:#ffffff;padding:18px;border-bottom:1px solid #e5e7eb;">
    <div style="font-size:14px;font-weight:800;color:#0f172a;margin-bottom:8px;">🚩 Risk Signals Detected</div>
    <ul style="margin:0;padding-left:20px;">{flags_li}</ul>
  </div>
  <div style="background:#ffffff;padding:18px;">
    <div style="font-size:14px;font-weight:800;color:#0f172a;margin-bottom:10px;">🤖 ML Model Signals ({ml_votes}/{model_count} flag catfish)</div>
    <table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
      <tr style="background:#1e293b;color:#f8fafc;">
        <th style="padding:10px 12px;font-size:12px;text-align:left;">Model</th>
        <th style="padding:10px 12px;font-size:12px;">Threshold</th>
        <th style="padding:10px 12px;font-size:12px;">Probability</th>
        <th style="padding:10px 12px;font-size:12px;">Verdict</th>
      </tr>
      {ml_rows}
    </table>
    <div style="font-size:12px;color:#334155;margin-top:10px;padding:10px;background:#f1f5f9;border-radius:8px;line-height:1.5;">
      ℹ️ <b>Ensemble note:</b> Six tuned models (class-conditional GMM + label-aware KMeans) provide ML probabilities.
      The blended score combines behavioral z-scores (65%) with ML consensus (35%).
    </div>
  </div>
</div>
"""


def render_scan_summary(raw_input: Dict[str, float], artifacts: DetectorArtifacts) -> Dict[str, Any]:
    result = scan_input(raw_input, artifacts)
    thresholds = _normalize_legacy_model_names(dict(artifacts.thresholds))
    model_probs = normalize_probability_map(result["model_probs"])
    model_details = build_model_details(model_probs, thresholds)
    ml_votes = sum(1 for row in model_details if row["flagged"])
    is_catfish = result["final_verdict"] == "CATFISH"
    verdict = "CATFISH DETECTED" if is_catfish else "LIKELY GENUINE"

    top_flags = [{"name": name, "value": float(value)} for name, value in result["top_flags"]]
    html_report = build_scan_html(
        behavioral_score=float(result["behavioral_score"]),
        is_catfish=is_catfish,
        ml_votes=ml_votes,
        model_count=len(model_details),
        raw_input=raw_input,
        top_flags=result["top_flags"],
        model_details=model_details,
    )

    result.update(
        {
            "verdict_label": verdict,
            "model_probs": model_probs,
            "thresholds": {row["name"]: row["threshold"] for row in model_details},
            "model_details": model_details,
            "ml_votes": ml_votes,
            "top_flags": top_flags,
            "html_report": html_report,
            "input_summary": raw_input,
        }
    )
    return result
