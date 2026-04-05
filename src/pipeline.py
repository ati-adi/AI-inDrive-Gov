import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(__file__).parent
ROOT = SRC.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

from data_loader import load_and_prepare
from synthetic_isj import generate_isj
from feature_engineering import build_features
from models.xgboost_model import train_model, save_model, load_model, has_saved_model
from models.anomaly_detector import detect_anomalies, detect_temporal_clusters
from scorer import compute_component_scores, calibrate_weights, compute_final_score, build_score_card
from rule_checker import check_rules


def run_pipeline(mode="with_isj", predict_only=False, verbose=True):
    t0 = time.time()
    results = {}

    if verbose: print("Step 1/8: Loading & cleaning data...")
    df = load_and_prepare()
    results["n_applications"] = len(df)
    if verbose: print(f"  -> {len(df)} applications loaded")

    if mode == "with_isj":
        if verbose: print("Step 2/8: Generating ISJ farmer profiles...")
        df, profiles = generate_isj(df)
        results["n_farmers"] = len(profiles)
        if verbose: print(f"  -> {len(profiles)} farmers created")
    else:
        profiles = None

    if verbose: print("Step 3/8: Engineering features...")
    X, y, feature_names, df_ml = build_features(df, mode=mode)
    results["n_features"] = len(feature_names)
    results["n_train_samples"] = len(X)
    if verbose: print(f"  -> {len(feature_names)} features, {len(X)} samples")

    if predict_only and has_saved_model():
        if verbose: print("Step 4/8: Loading saved model (predict only)...")
        model, saved_features, saved_weights = load_model()
        missing = [f for f in saved_features if f not in X.columns]
        extra = [f for f in X.columns if f not in saved_features]
        for f in missing:
            X[f] = 0
        X = X[[f for f in saved_features if f in X.columns]]
        results["xgb_auc"] = 0
        results["xgb_auc_std"] = 0
        if verbose: print(f"  -> Model loaded, predicting on {len(X)} samples")
    else:
        if verbose: print("Step 4/8: Training XGBoost...")
        has_labels = y.nunique() >= 2 and len(y) >= 10
        if has_labels:
            labeled_idx = y.index
            X_train = X.loc[labeled_idx]
            model, cv_scores, shap_vals, importance = train_model(X_train, y, mode=mode)
            results["xgb_auc"] = float(cv_scores.mean())
            results["xgb_auc_std"] = float(cv_scores.std())
            if verbose: print(f"  -> CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        else:
            if verbose: print("  -> Not enough labels, loading saved model...")
            model, saved_features, saved_weights = load_model()
            if model is None:
                if verbose: print("  -> No saved model. Using dummy scores.")
                df_ml["comp_xgb"] = 50
                model = None
            else:
                missing = [f for f in saved_features if f not in X.columns]
                for f in missing:
                    X[f] = 0
                X = X[[f for f in saved_features if f in X.columns]]
            results["xgb_auc"] = 0
            results["xgb_auc_std"] = 0
            importance = pd.DataFrame({"feature": feature_names, "importance": 0, "mean_shap": 0})

    if verbose: print("Step 5/8: Detecting anomalies...")
    df_ml, iso_model = detect_anomalies(df_ml)
    df_ml = detect_temporal_clusters(df_ml)
    results["n_anomalies"] = int(df_ml["is_anomaly"].sum())
    results["n_coordinated"] = int(df_ml["coordination_flag"].sum())
    if verbose: print(f"  -> {results['n_anomalies']} anomalies, {results['n_coordinated']} coordinated")

    if verbose: print("Step 5.5/8: Checking rules from PDF...")
    df_ml = check_rules(df_ml)
    results["n_violations"] = int((df_ml["rule_violation_count"] > 0).sum())
    if verbose: print(f"  -> {results['n_violations']} with violations")

    if verbose: print("Step 6/8: Computing component scores...")
    if model is not None:
        df_ml = compute_component_scores(df_ml, model, X)

    if verbose: print("Step 7/8: Calibrating weights...")
    has_labels_for_cal = "label" in df_ml.columns and df_ml["label"].isin([0,1]).sum() >= 10
    if has_labels_for_cal:
        weights, cal_auc, lr_model, scaler = calibrate_weights(df_ml)
        results["calibrated_auc"] = float(cal_auc)
        results["weights"] = {k: round(float(v), 4) for k, v in weights.items()}
        if not predict_only and model is not None:
            save_model(model, feature_names, results["weights"])
            if verbose: print("  -> Model + weights saved to disk")
    else:
        if has_saved_model():
            _, _, weights = load_model()
        else:
            comp_cols = [c for c in df_ml.columns if c.startswith("comp_")]
            weights = {c: 1.0/len(comp_cols) for c in comp_cols}
        results["calibrated_auc"] = 0
        results["weights"] = weights

    if verbose:
        for k, v in sorted(results["weights"].items(), key=lambda x: -x[1]):
            print(f"     {k:25s} {v:.4f}")

    if verbose: print("Step 8/8: Computing final scores...")
    df_ml = compute_final_score(df_ml, weights)

    rec_dist = df_ml["recommendation"].value_counts().to_dict()
    results["recommendation_distribution"] = {str(k): int(v) for k, v in rec_dist.items()}
    results["score_stats"] = {
        "mean": round(float(df_ml["final_score"].mean()), 1),
        "median": round(float(df_ml["final_score"].median()), 1),
    }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = f"{elapsed:.1f}s"
    if verbose:
        print(f"\nPipeline complete in {elapsed:.1f}s")
        print(f"Recommendations: {rec_dist}")

    return {
        "df_scored": df_ml,
        "profiles": profiles,
        "model": model,
        "importance": importance if 'importance' in dir() else pd.DataFrame(),
        "weights": weights,
        "results": results,
        "feature_names": feature_names,
    }


def save_results(pipeline_output, output_dir=None):
    if output_dir is None:
        output_dir = DATA_OUT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pipeline_output["df_scored"]
    export_cols = [
        "app_number", "district", "oblast", "direction_code", "subsidy_name",
        "amount", "rate", "unit_count", "status",
        "final_score", "recommendation", "recommendation_ru",
        "comp_xgb", "comp_anomaly", "comp_regional", "comp_volume", "comp_rules",
        "anomaly_score", "is_anomaly", "coordination_flag",
        "rule_violation_count",
    ]
    for col in ["reliability_score", "herd_trend", "years_active", "comp_reliability", "farmer_id",
                "is_new_farmer", "herd_declining",
                "expected_mortality", "actual_mortality", "mortality_deviation", "pasture_norm_ha"]:
        if col in df.columns:
            export_cols.append(col)

    available_cols = [c for c in export_cols if c in df.columns]
    df[available_cols].to_csv(output_dir / "scored_applications.csv", index=False)

    if pipeline_output.get("profiles") is not None:
        pipeline_output["profiles"].to_csv(output_dir / "farmer_profiles.csv", index=False)

    if not pipeline_output.get("importance", pd.DataFrame()).empty:
        pipeline_output["importance"].to_csv(output_dir / "feature_importance.csv", index=False)

    meta = pipeline_output["results"].copy()
    with open(output_dir / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    try:
        from database import init_db, save_applications_from_df
        init_db()
        n_saved = save_applications_from_df(df[available_cols])
        if True: print(f"  Database: {n_saved} applications saved")
    except Exception as e:
        print(f"  Database save failed: {e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "with_isj"
    output = run_pipeline(mode=mode)
    save_results(output)
