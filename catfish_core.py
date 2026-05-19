from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from imblearn.combine import SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


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


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dating_app_behavior_dataset.csv"
NOTEBOOK_PATH = BASE_DIR / "WIA1006_Catfish_Group7_V15_FIXED.ipynb"
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

EPS = 1e-6


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
    return input_frame.values


def train_models(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=42,
            verbosity=0,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            max_iter=500,
            random_state=42,
        ),
    }

    for model in models.values():
        model.fit(x_train, y_train)
    return models


def find_thresholds(models: Dict[str, Any], x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        precision, recall, threshold_values = precision_recall_curve(y_test, probabilities)
        best_threshold = 0.40
        best_f1 = -1.0
        for index, threshold in enumerate(threshold_values):
            if 0.35 <= threshold <= 0.75:
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
    a = float(raw_input.get("app_usage_time_min", 0.0))
    s = float(raw_input.get("swipe_right_ratio", 0.0))
    b = float(raw_input.get("bio_length", 0.0))
    m = float(raw_input.get("message_sent_count", 0.0))
    pics = float(raw_input.get("profile_pics_count", 0.0))
    likes = float(raw_input.get("likes_received", 0.0))
    matches = float(raw_input.get("mutual_matches", 0.0))

    def zscore(column: str, value: float) -> float:
        mean, std = population_stats.get(column, (0.0, 1.0))
        return (value - mean) / (std + EPS)

    engagement = m / (a + 1)
    base_engagement = population_stats.get("message_sent_count", (1.0, 1.0))[0] / (
        population_stats.get("app_usage_time_min", (1.0, 1.0))[0] + 1
    )

    likes_mean, likes_std = population_stats.get("likes_received", (1.0, 1.0))
    matches_mean, matches_std = population_stats.get("mutual_matches", (1.0, 1.0))
    likes_match_base = likes_mean / (matches_mean + 1)
    likes_match_current = likes / (matches + 1)

    # Increase sensitivity so extreme values can push the behavioral score
    # closer to the 99-100% range when inputs are highly anomalous.
    # Allow component scores to exceed 1.0 for extreme z-scores so the
    # combined weighted score can approach 100. We only clip negative values.
    component_scores = {
        "High message count": max(0.0, zscore("message_sent_count", m)) / 1.0,
        "Extreme swipe pattern": max(0.0, abs(zscore("swipe_right_ratio", s))) / 1.0,
        "Suspiciously short bio": max(0.0, -zscore("bio_length", b)) / 1.0,
        "Overlong bio": max(0.0, zscore("bio_length", b)) / 1.0,
        "High engagement density": max(0.0, (engagement - base_engagement) / (base_engagement + EPS)) / 0.6,
        "Very few profile pics": max(0.0, -zscore("profile_pics_count", pics)) / 0.6,
        "High likes, few matches": max(0.0, (likes_match_current - likes_match_base) / (likes_match_base + EPS)) / 0.6,
        "Excessive app usage": max(0.0, zscore("app_usage_time_min", a)) / 1.0,
        "Very few mutual matches": max(0.0, -zscore("mutual_matches", matches)) / 1.0,
    }

    component_weights = {
        "High message count": 0.20,
        "Extreme swipe pattern": 0.12,
        "Suspiciously short bio": 0.18,
        "Overlong bio": 0.04,
        "High engagement density": 0.14,
        "Very few profile pics": 0.08,
        "High likes, few matches": 0.10,
        "Excessive app usage": 0.08,
        "Very few mutual matches": 0.06,
    }

    # Each component contributes score*weight*100; allow exceeding 100 before final clamp
    risk_components = {name: component_scores[name] * component_weights.get(name, 0.0) * 100.0 for name in component_scores}
    weighted = sum(risk_components.values())
    risk = round(min(100.0, max(0.0, weighted)), 1)
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

    model_probs = {name: float(model.predict_proba(vector)[0][1]) for name, model in artifacts.models.items()}
    ml_votes = sum(1 for name, prob in model_probs.items() if prob >= artifacts.thresholds.get(name, 0.40))
    behavioral_score, top_flags = behavioral_risk(raw_input, artifacts.population_stats)
    final_verdict = "CATFISH" if behavioral_score >= 30.0 else "GENUINE"

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

    train_resampled, y_train_resampled = SMOTETomek(random_state=42).fit_resample(x_train_arr, y_train_arr)

    models = train_models(train_resampled, y_train_resampled)
    # Wrap estimators in a lightweight callable wrapper so the saved artifacts
    # expose consistent `predict_proba` and `predict` methods and remain pickleable.
    models = {name: ModelFunction(m) for name, m in models.items()}
    thresholds = find_thresholds(models, x_test_arr, y_test_arr)

    importance_model = ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    importance_model.fit(train_resampled, y_train_resampled)
    feature_importances = dict(zip(feature_names, importance_model.feature_importances_))

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
            sns.barplot(x=list(vals), y=list(names), palette="viridis")
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
            sns.barplot(x="F1-Score", y="Model", data=lb.sort_values("F1-Score", ascending=False), palette="magma")
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


def load_artifacts() -> DetectorArtifacts:
    if ARTIFACT_BUNDLE_PATH.exists():
        return joblib.load(ARTIFACT_BUNDLE_PATH)

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


def render_scan_summary(raw_input: Dict[str, float], artifacts: DetectorArtifacts) -> Dict[str, Any]:
    result = scan_input(raw_input, artifacts)
    verdict = "CATFISH DETECTED" if result["final_verdict"] == "CATFISH" else "LIKELY GENUINE"
    result["verdict_label"] = verdict
    return result