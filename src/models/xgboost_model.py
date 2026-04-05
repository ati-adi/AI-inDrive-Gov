import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from pathlib import Path
import shap
import json

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def save_model(model, feature_names, weights=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgboost.json"))
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(list(feature_names), f)
    if weights:
        with open(MODEL_DIR / "weights.json", "w") as f:
            json.dump(weights, f)


def load_model():
    path = MODEL_DIR / "xgboost.json"
    if not path.exists():
        return None, None, None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    features = []
    if (MODEL_DIR / "features.json").exists():
        with open(MODEL_DIR / "features.json") as f:
            features = json.load(f)
    weights = {}
    if (MODEL_DIR / "weights.json").exists():
        with open(MODEL_DIR / "weights.json") as f:
            weights = json.load(f)
    return model, features, weights


def has_saved_model():
    return (MODEL_DIR / "xgboost.json").exists()


def train_model(X, y, mode="with_isj"):
    scale = (y == 1).sum() / max((y == 0).sum(), 1)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "scale_pos_weight": 1 / max(scale, 0.01),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
        "mean_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_shap", ascending=False)
    return model, cv_scores, shap_values, importance


def predict_with_explanation(model, X_single, feature_names):
    explainer = shap.TreeExplainer(model)
    if isinstance(X_single, pd.Series):
        X_single = X_single.to_frame().T
    proba = model.predict_proba(X_single)[0, 1]
    sv = explainer.shap_values(X_single)[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    reasons = []
    for i in top_idx:
        direction = "повышает" if sv[i] > 0 else "снижает"
        reasons.append({
            "feature": feature_names[i],
            "value": float(X_single.iloc[0, i]),
            "shap": float(sv[i]),
            "direction": direction,
        })
    return {"approval_probability": float(proba), "top_reasons": reasons}
