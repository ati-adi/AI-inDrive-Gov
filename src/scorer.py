"""
scorer.py — собирает все сигналы в финальный балл 0-100.

Компоненты:
  1. XGBoost probability (supervised, data-driven)
  2. Anomaly score (Isolation Forest)
  3. Rule compliance (из PDF правил)
  4. Regional factor (район vs среднее)
  5. Volume/Impact (масштаб хозяйства)

Веса калибруются через Logistic Regression на реальных labels.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


def compute_component_scores(df, xgb_model, X_features):
    df = df.copy()
    if xgb_model is not None and len(X_features) > 0:
        try:
            proba = xgb_model.predict_proba(X_features)[:, 1]
            if len(proba) == len(df):
                df["comp_xgb"] = proba * 100
            else:
                df["comp_xgb"] = 50.0
        except Exception:
            df["comp_xgb"] = 50.0
    else:
        df["comp_xgb"] = 50.0
    if "anomaly_score" in df.columns:
        df["comp_anomaly"] = (1 - df["anomaly_score"]) * 100
    else:
        df["comp_anomaly"] = 50.0
    if "comp_rules" not in df.columns:
        df["comp_rules"] = 70.0
    if "district_approve_rate" in df.columns:
        df["comp_regional"] = (df["district_approve_rate"] * 100).clip(0, 100)
    else:
        df["comp_regional"] = 50.0
    if "unit_count" in df.columns:
        uc = df["unit_count"].clip(lower=1)
        p5, p95 = uc.quantile(0.05), uc.quantile(0.95)
        denom = np.log1p(p95) - np.log1p(p5)
        if denom > 0:
            df["comp_volume"] = ((np.log1p(uc) - np.log1p(p5)) / denom * 100).clip(0, 100)
        else:
            df["comp_volume"] = 50.0
    else:
        df["comp_volume"] = 50.0
    if "reliability_score" in df.columns:
        df["comp_reliability"] = (df["reliability_score"] * 100).clip(0, 100)
    return df


def calibrate_weights(df, label_col="label"):
    comp_cols = [c for c in df.columns if c.startswith("comp_")]
    mask = df[label_col].isin([0, 1]) if label_col in df.columns else pd.Series([False]*len(df))
    if mask.sum() < 10:
        weights = {c: 1.0/len(comp_cols) for c in comp_cols}
        return weights, 0.0, None, None
    X = df.loc[mask, comp_cols].fillna(50)
    y = df.loc[mask, label_col].astype(int)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_scaled, y)
    raw_weights = np.abs(lr.coef_[0])
    weights = raw_weights / raw_weights.sum()
    weight_dict = dict(zip(comp_cols, weights))
    y_pred = lr.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return weight_dict, auc, lr, scaler


def compute_final_score(df, weights=None):
    """
    Считает финальный балл 0-100.
    
    Если weights=None, использует равные веса.
    """
    df = df.copy()
    comp_cols = [c for c in df.columns if c.startswith("comp_")]
    
    if weights is None:
        weights = {c: 1.0 / len(comp_cols) for c in comp_cols}
    
    # Weighted sum
    score = sum(df[col].fillna(50) * weights.get(col, 0) for col in comp_cols)
    df["final_score"] = score.round(1)
    
    # Бинарная рекомендация: одобрить / отклонить
    THRESHOLD = 50
    df["recommendation"] = np.where(
        df["final_score"] >= THRESHOLD,
        "approve",
        "reject"
    )
    df["recommendation_ru"] = np.where(
        df["final_score"] >= THRESHOLD,
        "Рекомендуется к одобрению",
        "Рекомендуется к отклонению"
    )
    
    return df


def build_score_card(row, weights):
    comp_cols = [c for c in row.index if c.startswith("comp_")]
    components = []
    for col in comp_cols:
        name = col.replace("comp_", "").replace("_", " ").title()
        value = float(row[col]) if pd.notna(row[col]) else 50.0
        weight = float(weights.get(col, 0))
        contribution = value * weight
        components.append({
            "name": name,
            "score": round(value, 1),
            "weight": round(weight, 3),
            "contribution": round(contribution, 1),
        })
    components.sort(key=lambda x: x["contribution"], reverse=True)
    return {
        "final_score": round(float(row.get("final_score", 0)), 1),
        "recommendation": str(row.get("recommendation", "")),
        "recommendation_ru": str(row.get("recommendation_ru", "")),
        "components": components,
        "district": str(row.get("district", "")),
        "direction": str(row.get("direction_code", "")),
        "amount": float(row.get("amount", 0)),
        "unit_count": float(row.get("unit_count", 0)),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_and_prepare
    from synthetic_isj import generate_isj
    from feature_engineering import build_features
    from models.xgboost_model import train_model
    from models.anomaly_detector import detect_anomalies, detect_temporal_clusters
    
    print("=== FULL PIPELINE ===\n")
    
    # Step 1: Load & clean
    print("1. Loading data...")
    df = load_and_prepare()
    
    # Step 2: ISJ
    print("2. Generating ISJ profiles...")
    df, profiles = generate_isj(df)
    
    # Step 3: Features
    print("3. Building features...")
    X, y, feature_names, df_ml = build_features(df, mode="with_isj")
    
    # Step 4: XGBoost
    print("4. Training XGBoost...")
    model, cv_scores, shap_vals, importance = train_model(X, y)
    print(f"   CV AUC: {cv_scores.mean():.4f}")
    
    # Step 5: Anomaly detection (на полном df, не только ml)
    print("5. Detecting anomalies...")
    df_ml, _ = detect_anomalies(df_ml)
    df_ml = detect_temporal_clusters(df_ml)
    
    # Step 6: Component scores
    print("6. Computing component scores...")
    df_ml = compute_component_scores(df_ml, model, X)
    
    # Step 7: Calibrate weights
    print("7. Calibrating weights...")
    weights, cal_auc, lr_model, scaler = calibrate_weights(df_ml)
    print(f"   Calibrated AUC: {cal_auc:.4f}")
    print(f"   Weights: {weights}")
    
    # Step 8: Final scores
    print("8. Computing final scores...")
    df_ml = compute_final_score(df_ml, weights)
    
    print(f"\n=== RESULTS ===")
    print(f"Score distribution:")
    print(df_ml["final_score"].describe())
    print(f"\nRisk categories:")
    print(df_ml["risk_category"].value_counts())
    
    # Sample score card
    print(f"\n=== SAMPLE SCORE CARD ===")
    sample = df_ml.iloc[100]
    card = build_score_card(sample, weights)
    print(f"Score: {card['final_score']} ({card['risk_category']})")
    print(f"District: {card['district']}, Direction: {card['direction']}")
    print(f"Amount: {card['amount']:,.0f} tg, Units: {card['unit_count']}")
    print(f"Recommendation: {card['recommendation']}")
    for comp in card["components"]:
        print(f"  {comp['name']:20s} score={comp['score']:5.1f}  weight={comp['weight']:.3f}  contrib={comp['contribution']:5.1f}")
