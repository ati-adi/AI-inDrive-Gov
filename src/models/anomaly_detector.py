"""
anomaly_detector.py — Isolation Forest + Temporal Clustering.

Два независимых сигнала:
1. Isolation Forest — статистические выбросы по фичам
2. Temporal Clustering — координированные заявки (район + время + сумма)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# ---------- Isolation Forest ----------

ANOMALY_FEATURES = [
    "unit_count", "amount", "rate",
    "amount_vs_district", "units_vs_district",
    "submit_hour",
]


def detect_anomalies(df, contamination=0.05):
    df = df.copy()
    available = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[available].fillna(0)
    if len(X) < 5:
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = 0
        return df, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=42,
    )
    model.fit(X_scaled)
    raw_scores = model.score_samples(X_scaled)
    score_range = raw_scores.max() - raw_scores.min()
    if score_range > 0:
        df["anomaly_score"] = 1 - (raw_scores - raw_scores.min()) / score_range
    else:
        df["anomaly_score"] = 0.0
    df["is_anomaly"] = (model.predict(X_scaled) == -1).astype(int)
    return df, model


# ---------- Temporal Clustering ----------

def detect_temporal_clusters(df, time_window_minutes=30, min_cluster_size=5):
    """
    Ищет координированные заявки: одинаковый район + тип субсидии,
    близкое время (30 мин), похожие суммы.
    
    Жёсткие параметры: 30 минут + минимум 5 заявок + одинаковый тип.
    """
    df = df.copy()
    df["coordination_flag"] = 0
    df["cluster_id"] = -1
    df["cluster_size"] = 0
    
    t_min = df["submitted_at"].min()
    df["t_minutes"] = (df["submitted_at"] - t_min).dt.total_seconds() / 60
    
    cluster_counter = 0
    
    # Группируем по район + тип субсидии (жёстче чем просто район)
    for (district, direction), group in df.groupby(["district", "direction_code"]):
        if len(group) < min_cluster_size:
            continue
        
        # Нормализуем: время в единицах time_window, сумма в std
        X_cluster = np.column_stack([
            group["t_minutes"].values / time_window_minutes,
            (group["amount"].values - group["amount"].mean()) / (group["amount"].std() + 1e-6) * 0.3,
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=min_cluster_size)
        labels = dbscan.fit_predict(X_cluster)
        
        for label in set(labels):
            if label == -1:
                continue
            
            mask = labels == label
            indices = group.index[mask]
            size = mask.sum()
            
            # Дополнительный фильтр: суммы должны быть похожи (cv < 0.3)
            amounts = df.loc[indices, "amount"]
            cv = amounts.std() / (amounts.mean() + 1e-6)
            
            if size >= min_cluster_size and cv < 0.3:
                df.loc[indices, "coordination_flag"] = 1
                df.loc[indices, "cluster_id"] = cluster_counter
                df.loc[indices, "cluster_size"] = size
                cluster_counter += 1
    
    df = df.drop(columns=["t_minutes"], errors="ignore")
    
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data_loader import load_and_prepare
    
    df = load_and_prepare()
    
    # Isolation Forest
    df, iso_model = detect_anomalies(df)
    print("=== Isolation Forest ===")
    print(f"Anomalies detected: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")
    print(f"Anomaly score stats:\n{df['anomaly_score'].describe()}")
    
    # Проверяем: аномалии чаще отклоняются?
    anomaly_reject = df[df["is_anomaly"]==1]["label"].value_counts(normalize=True)
    normal_reject = df[df["is_anomaly"]==0]["label"].value_counts(normalize=True)
    print(f"\nReject rate among anomalies: {anomaly_reject.get(0, 0):.1%}")
    print(f"Reject rate among normals:   {normal_reject.get(0, 0):.1%}")
    
    # Temporal Clustering
    print("\n=== Temporal Clustering ===")
    df = detect_temporal_clusters(df)
    print(f"Coordination flags: {df['coordination_flag'].sum()} ({df['coordination_flag'].mean():.1%})")
    print(f"Unique clusters: {df[df['cluster_id']>=0]['cluster_id'].nunique()}")
    print(f"Avg cluster size: {df[df['cluster_size']>0]['cluster_size'].mean():.1f}")
    
    # Координированные заявки чаще отклоняются?
    coord_reject = df[df["coordination_flag"]==1]["label"].value_counts(normalize=True)
    nocoord_reject = df[df["coordination_flag"]==0]["label"].value_counts(normalize=True)
    print(f"\nReject rate in coordinated: {coord_reject.get(0, 0):.1%}")
    print(f"Reject rate in normal:     {nocoord_reject.get(0, 0):.1%}")
