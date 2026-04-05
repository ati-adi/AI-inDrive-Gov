"""
feature_engineering.py — собирает фичи из Excel + ИСЖ в ML-ready формат.

Два режима:
  - without_isj: только фичи из Excel (всегда доступны)
  - with_isj: + фичи из синтетического/реального ИСЖ
"""

import pandas as pd
import numpy as np

# Фичи которые ВСЕГДА доступны (из Excel)
BASE_FEATURES = [
    "unit_count",           # количество голов (сумма/норматив)
    "amount",               # причитающая сумма
    "rate",                 # норматив
    "submit_hour",          # час подачи
    "submit_dow",           # день недели
    "submit_month",         # месяц
    "amount_vs_district",   # сумма vs среднее по району
    "units_vs_district",    # головы vs среднее по району
    "district_app_count",   # сколько заявок в районе
    "district_approve_rate",# approve rate района
]

# Фичи которые добавляются с ИСЖ
ISJ_FEATURES = [
    "years_active",         # сколько лет фермер активен
    "herd_size_prev_year",  # стадо в прошлом году
    "past_applications",    # прошлые заявки
    "past_approve_rate",    # % одобренных в прошлом
    "reliability_score",    # композитный балл надёжности
    "is_new_farmer",        # первый раз подаёт
    "herd_declining",       # стадо падает
    "expected_mortality",   # норма падежа для данного типа
    "actual_mortality",     # фактический падёж
    "mortality_deviation",  # отклонение от нормы (>0 = плохо)
    "pasture_norm_ha",      # норма га/голова для области
]

# Категориальные фичи для one-hot encoding
CATEGORICAL_FEATURES = [
    "direction_code",       # тип субсидии (cattle, poultry, etc)
]


def build_features(df, mode="with_isj"):
    """
    Собирает финальный набор фичей.
    
    mode: "with_isj" | "without_isj"
    
    Возвращает (X, y, feature_names)
    """
    df = df.copy()
    
    # Базовые фичи (всегда)
    feature_cols = BASE_FEATURES.copy()
    
    # ИСЖ фичи (если доступны)
    if mode == "with_isj":
        available_isj = [f for f in ISJ_FEATURES if f in df.columns]
        feature_cols += available_isj
    
    # One-hot для категориальных
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            feature_cols += list(dummies.columns)
    
    # Herd growth ratio (если ИСЖ)
    if mode == "with_isj" and "herd_size_current" in df.columns and "herd_size_prev_year" in df.columns:
        df["herd_growth_ratio"] = df["herd_size_current"] / df["herd_size_prev_year"].replace(0, 1)
        feature_cols.append("herd_growth_ratio")
    
    # Log-transform для сильно скошенных фичей
    for col in ["amount", "unit_count", "herd_size_current", "herd_size_prev_year"]:
        if col in df.columns:
            log_col = f"log_{col}"
            df[log_col] = np.log1p(df[col].clip(lower=0))
            feature_cols.append(log_col)
    
    mask = df["label"].isin([0, 1])
    df_ml = df.copy()
    X = df_ml[feature_cols].fillna(0).astype(float)
    if mask.sum() >= 2:
        y = df_ml.loc[mask, "label"].astype(int)
    else:
        y = pd.Series(dtype=int)

    return X, y, feature_cols, df_ml


if __name__ == "__main__":
    from data_loader import load_and_prepare
    from synthetic_isj import generate_isj
    
    df = load_and_prepare()
    df_enriched, profiles = generate_isj(df)
    
    # Режим БЕЗ ИСЖ
    X_base, y_base, cols_base, _ = build_features(df, mode="without_isj")
    print(f"=== WITHOUT ISJ ===")
    print(f"X shape: {X_base.shape}")
    print(f"Features ({len(cols_base)}): {cols_base}")
    print(f"y: {y_base.value_counts().to_dict()}")
    
    print()
    
    # Режим С ИСЖ
    X_full, y_full, cols_full, _ = build_features(df_enriched, mode="with_isj")
    print(f"=== WITH ISJ ===")
    print(f"X shape: {X_full.shape}")
    print(f"Features ({len(cols_full)}): {cols_full}")
    print(f"y: {y_full.value_counts().to_dict()}")
