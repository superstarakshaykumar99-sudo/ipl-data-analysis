"""
model_training.py
-----------------
Train, evaluate, and persist an IPL match winner predictor.
"""

import os
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "match_winner_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

FEATURE_COLS = ["team1", "team2", "toss_winner", "toss_decision", "venue"]


def prepare_features(matches: pd.DataFrame) -> tuple:
    """Prepare features, labels, and fitted LabelEncoders for model training.

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        Tuple of (X DataFrame, y Series, encoders dict)
    """
    df = matches.copy()
    df = df.dropna(subset=["winner"] + FEATURE_COLS)

    encoders = {}
    for col in FEATURE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df["target"] = target_le.fit_transform(df["winner"].astype(str))
    encoders["winner"] = target_le

    X = df[FEATURE_COLS]
    y = df["target"]
    logger.info("Features prepared. X shape: %s, classes: %d", X.shape, len(target_le.classes_))
    return X, y, encoders


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train multiple classifiers and return the best one by CV accuracy.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (best_model, best_model_name, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    best_model = None
    best_name = ""
    best_cv = 0.0
    all_metrics = {}

    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        logger.info("%s | CV: %.4f ± %.4f | Test Acc: %.4f", name, cv_scores.mean(), cv_scores.std(), test_acc)
        all_metrics[name] = {
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(test_acc), 4),
        }

        if cv_scores.mean() > best_cv:
            best_cv = cv_scores.mean()
            best_model = clf
            best_name = name

    logger.info("Best model: %s (CV Accuracy: %.4f)", best_name, best_cv)
    print(f"\n=== Best Model: {best_name} ===")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    return best_model, best_name, all_metrics


def plot_feature_importance(model, feature_names: list, save: bool = True) -> None:
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(importance / importance.max())
    ax.barh(feature_names, importance, color=colors)
    ax.set_title("Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        charts_dir = os.path.join(BASE_DIR, "..", "images", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        fig.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_model(model, encoders: dict, model_name: str, metrics: dict) -> None:
    """Save the trained model, encoders, and metadata to disk.

    Args:
        model: Fitted sklearn model
        encoders: Dict of LabelEncoder objects keyed by column name
        model_name: Name of the winning model
        metrics: Metrics dict from train_model()
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", MODEL_PATH)

    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    logger.info("Encoders saved → %s", ENCODERS_PATH)

    metadata = {
        "model_name": model_name,
        "features": FEATURE_COLS,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", METADATA_PATH)


def load_model(path: str = MODEL_PATH):
    """Load the persisted model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_encoders(path: str = ENCODERS_PATH) -> dict:
    """Load the persisted label encoders from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_winner(team1: str, team2: str, toss_winner: str,
                   toss_decision: str, venue: str,
                   model=None, encoders: dict = None) -> str:
    """Predict the match winner given match parameters.

    Args:
        team1: First team name
        team2: Second team name
        toss_winner: Team that won the toss
        toss_decision: 'bat' or 'field'
        venue: Stadium name
        model: Pre-loaded model (loads from disk if None)
        encoders: Pre-loaded encoders dict (loads from disk if None)

    Returns:
        Predicted winner team name string
    """
    if model is None:
        model = load_model()
    if encoders is None:
        encoders = load_encoders()

    def safe_encode(le: LabelEncoder, value: str) -> int:
        """Encode value; return 0 if unseen."""
        if value in le.classes_:
            return int(le.transform([value])[0])
        return 0

    row = {
        "team1": safe_encode(encoders["team1"], team1),
        "team2": safe_encode(encoders["team2"], team2),
        "toss_winner": safe_encode(encoders["toss_winner"], toss_winner),
        "toss_decision": safe_encode(encoders["toss_decision"], toss_decision),
        "venue": safe_encode(encoders["venue"], venue),
    }
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    winner = encoders["winner"].inverse_transform([pred])[0]
    return winner


if __name__ == "__main__":
    matches = pd.read_csv("data/matches.csv")
    X, y, encoders = prepare_features(matches)
    model, model_name, metrics = train_model(X, y)
    plot_feature_importance(model, FEATURE_COLS)
    save_model(model, encoders, model_name, metrics)
    print("\nTest prediction:", predict_winner(
        "Mumbai Indians", "Chennai Super Kings",
        "Mumbai Indians", "bat", "Wankhede Stadium"
    ))
