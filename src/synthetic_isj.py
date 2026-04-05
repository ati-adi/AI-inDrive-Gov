"""
synthetic_isj.py — генерирует реалистичные профили фермеров из реального Excel.

Логика:
1. Группируем заявки в "фермеров" по (район, неделя подачи, совместимые типы)
2. Для каждого фермера вычисляем профиль из его заявок
3. Генерируем историю (прошлые годы) на основе распределений района

При подключении реального ИСЖ — заменяем этот файл на адаптер к API.
"""

import pandas as pd
import numpy as np
from pathlib import Path


SEED = 42


def assign_farmer_ids(df):
    """
    Группирует заявки в синтетических фермеров.
    
    Логика: заявки из одного района, поданные в одну неделю,
    с совместимыми типами субсидий — скорее всего один фермер.
    """
    rng = np.random.RandomState(SEED)
    
    # Неделя подачи как ключ группировки
    df = df.copy()
    df["submit_week"] = df["submitted_at"].dt.isocalendar().week.astype(int)
    
    # Группируем по (район, неделя, направление)
    # Внутри группы — каждая заявка может быть от одного фермера
    # или от разных. Используем вероятностную модель:
    # - маленькие группы (1-3 заявки) → скорее всего один фермер
    # - большие группы → разбиваем на подгруппы по 1-4 заявки
    
    farmer_ids = pd.Series(index=df.index, dtype=str)
    farmer_counter = 0
    
    groups = df.groupby(["district", "submit_week", "direction_code"])
    
    for (district, week, direction), group_df in groups:
        n = len(group_df)
        
        if n <= 3:
            # Маленькая группа — один фермер
            farmer_id = f"F{farmer_counter:06d}"
            farmer_ids.loc[group_df.index] = farmer_id
            farmer_counter += 1
        else:
            # Разбиваем на подгруппы по 1-4 заявки
            indices = group_df.index.tolist()
            rng.shuffle(indices)
            
            i = 0
            while i < len(indices):
                chunk_size = rng.randint(1, min(5, len(indices) - i + 1))
                farmer_id = f"F{farmer_counter:06d}"
                for idx in indices[i:i + chunk_size]:
                    farmer_ids.loc[idx] = farmer_id
                farmer_counter += 1
                i += chunk_size
    
    df["farmer_id"] = farmer_ids
    return df, farmer_counter


def build_farmer_profiles(df):
    """
    Строит профиль каждого фермера из его заявок.
    Всё что можно вычислить — вычисляем. Остальное генерируем.
    """
    rng = np.random.RandomState(SEED + 1)
    
    # Агрегируем заявки по фермеру
    profiles = df.groupby("farmer_id").agg(
        district=("district", "first"),
        oblast=("oblast", "first"),
        direction=("direction_code", "first"),
        num_applications=("app_number", "count"),
        total_amount=("amount", "sum"),
        total_units=("unit_count", "sum"),
        avg_rate=("rate", "mean"),
        approval_rate=("label", lambda x: (x == 1).mean()),
        has_rejection=("label", lambda x: (x == 0).any()),
        submit_hour_avg=("submit_hour", "mean"),
    ).reset_index()
    
    n = len(profiles)
    
    # --------- Генерируем ИСТОРИЧЕСКИЕ данные ---------
    # Эти поля НЕВОЗМОЖНО вычислить из Excel → синтетика
    # Но привязаны к реальным распределениям
    
    # Сколько лет фермер активен (1-15, Пуассон λ=4)
    profiles["years_active"] = np.clip(rng.poisson(4, n), 1, 15)
    
    # Стадо в прошлом году (текущее * случайный множитель)
    # Множитель зависит от района: в развитых — рост, в отстающих — стагнация
    district_growth = df.groupby("district")["district_approve_rate"].first()
    profiles["_district_health"] = profiles["district"].map(district_growth).fillna(0.5)
    
    # Хорошие районы (approve > 0.6) — стадо скорее растёт
    # Плохие районы (approve < 0.3) — стадо скорее падает
    growth_mean = 0.85 + profiles["_district_health"] * 0.3  # от 0.85 до 1.15
    growth_std = 0.15
    growth_factor = rng.normal(growth_mean, growth_std)
    growth_factor = np.clip(growth_factor, 0.5, 1.8)
    
    profiles["herd_size_current"] = profiles["total_units"]
    profiles["herd_size_prev_year"] = (profiles["total_units"] / growth_factor).round(0)
    
    # Тренд стада
    ratio = profiles["herd_size_current"] / profiles["herd_size_prev_year"].replace(0, 1)
    profiles["herd_trend"] = pd.cut(
        ratio,
        bins=[0, 0.85, 1.05, float("inf")],
        labels=["declining", "stable", "growing"]
    )
    
    # История прошлых заявок (привязана к years_active)
    profiles["past_applications"] = (profiles["years_active"] * rng.uniform(0.5, 2.0, n)).astype(int).clip(0)
    
    # Прошлый approve rate (привязан к текущему + шум)
    profiles["past_approve_rate"] = np.clip(
        profiles["approval_rate"] + rng.normal(0, 0.1, n),
        0, 1
    ).round(2)
    
    # --------- Reliability Score (композитный) ---------
    # Формула: 60% от approve history + 25% от herd trend + 15% от years active
    trend_map = {"growing": 1.0, "stable": 0.6, "declining": 0.2}
    trend_score = profiles["herd_trend"].astype(str).map(trend_map).fillna(0.5)
    
    years_score = np.clip(profiles["years_active"] / 10, 0, 1)
    
    profiles["reliability_score"] = (
        0.60 * profiles["past_approve_rate"] +
        0.25 * trend_score +
        0.15 * years_score
    ).round(3)
    
    # --------- Флаги ---------
    profiles["is_new_farmer"] = (profiles["years_active"] <= 1).astype(int)
    profiles["herd_declining"] = (profiles["herd_trend"] == "declining").astype(int)
    
    # --------- Нормы падежа (из PDF) ---------
    from rule_checker import get_mortality_norm, get_pasture_norm
    
    profiles["expected_mortality"] = [
        get_mortality_norm(d, "") for d in profiles["direction"]
    ]
    
    # Фактический падёж: ожидаемый + шум (хорошие фермеры ниже нормы)
    actual_mort = profiles["expected_mortality"] + rng.normal(0, 0.01, n)
    # Плохие фермеры (low reliability) — выше нормы
    actual_mort += (1 - profiles["reliability_score"]) * 0.02
    profiles["actual_mortality"] = np.clip(actual_mort, 0, 0.3).round(3)
    
    # Отклонение от нормы (>0 = выше нормы = плохо)
    profiles["mortality_deviation"] = (
        profiles["actual_mortality"] - profiles["expected_mortality"]
    ).round(3)
    
    # --------- Пастбищная нагрузка ---------
    profiles["pasture_norm_ha"] = [
        get_pasture_norm(o) for o in profiles["oblast"]
    ]
    
    # Убираем служебные колонки
    profiles = profiles.drop(columns=["_district_health"], errors="ignore")
    
    return profiles


def merge_isj_with_applications(df, profiles):
    """
    Присоединяет профиль фермера обратно к заявкам.
    Это финальный датасет для скоринга.
    """
    isj_cols = [
        "farmer_id", "years_active", "herd_size_current", "herd_size_prev_year",
        "herd_trend", "past_applications", "past_approve_rate",
        "reliability_score", "is_new_farmer", "herd_declining",
        "expected_mortality", "actual_mortality", "mortality_deviation",
        "pasture_norm_ha",
    ]
    return df.merge(profiles[isj_cols], on="farmer_id", how="left")


def generate_isj(df):
    """
    Полный пайплайн генерации ИСЖ.
    Возвращает (df_enriched, profiles).
    """
    df, n_farmers = assign_farmer_ids(df)
    profiles = build_farmer_profiles(df)
    df_enriched = merge_isj_with_applications(df, profiles)
    return df_enriched, profiles


if __name__ == "__main__":
    from data_loader import load_and_prepare
    
    df = load_and_prepare()
    df_enriched, profiles = generate_isj(df)
    
    print(f"Applications: {len(df_enriched)}")
    print(f"Unique farmers: {len(profiles)}")
    print(f"\nFarmer profile sample:")
    print(profiles[["farmer_id","district","direction","num_applications",
                     "herd_size_current","herd_trend","reliability_score",
                     "years_active","is_new_farmer"]].head(15).to_string())
    
    print(f"\nHerd trend distribution:")
    print(profiles["herd_trend"].value_counts())
    
    print(f"\nReliability score stats:")
    print(profiles["reliability_score"].describe())
    
    print(f"\nNew farmers: {profiles['is_new_farmer'].sum()} ({profiles['is_new_farmer'].mean():.1%})")
    print(f"Declining herds: {profiles['herd_declining'].sum()} ({profiles['herd_declining'].mean():.1%})")
