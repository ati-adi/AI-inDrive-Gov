"""
rule_checker.py — проверяет заявки по правилам из PDF (Приложение 1 и 2).

Источники правил:
  - Приложение 1: нормативы субсидий (ставки)
  - Приложение 2: критерии (возраст, соотношения, сроки, минимумы)
  - Нормы падежа: допустимые % убыли по типам животных
  - Нормы пастбищ: максимальная нагрузка на га по регионам
"""

import pandas as pd
import numpy as np


# ========== НОРМАТИВЫ ИЗ ПРИЛОЖЕНИЯ 1 ==========

VALID_RATES_BY_DIRECTION = {
    "cattle": {260000,150000,390000,525000,350000,700000,15000,300,175,200,
               45,30,20,80000,10000,5000,1500,25194,6298,219000,160000,270000,
               200000,225000,400000,300000,40000},
    "poultry": {600,80,70,60,50},
    "sheep": {26000,15000,52000,260000,4000,3000,7000,80000,1500,200,150,25},
    "horse": {20000,175000,100000,858500,40000},
    "camel": {175000},
    "pig": {100000,2000},
    "goat": {70000},
    "insemination": {5000,1500},
    "beekeeping": {200},
}

# ========== ПРАВИЛА ВОЗРАСТА ИЗ ПРИЛОЖЕНИЯ 2 ==========
# Формат: keyword_in_subsidy_name -> (min_months, max_months, description)
AGE_RULES = {
    "быков-производителей мясных": (8, 26, "Быки мясные 8-26 мес"),
    "маточного поголовья крупного рогатого скота": (6, 26, "Маточное КРС 6-26 мес"),
    "племенных овец": (4, 18, "Овцы 4-18 мес"),
    "барана-производителя": (4, 18, "Бараны 4-18 мес"),
    "жеребца-производителя": (18, 60, "Жеребцы 18-60 мес"),
    "верблюдов-производителей": (12, 60, "Верблюды 12-60 мес"),
    "поголовья свиней": (3, 12, "Свиньи 3-12 мес"),
    "маточного поголовья коз": (6, 18, "Козы 6-18 мес"),
}

# ========== СООТНОШЕНИЯ ПРОИЗВОДИТЕЛЬ/МАТКИ ==========
RATIO_RULES = {
    "быков-производителей": {"min_females": 20, "max_females": 30, "desc": "20-30 маток на 1 быка"},
    "барана-производителя": {"min_females": 20, "max_females": 30, "desc": "20-30 маток на 1 барана"},
    "жеребца-производителя": {"min_females": 20, "max_females": 30, "desc": "20-30 маток на жеребца"},
    "верблюдов-производителей": {"min_females": 20, "max_females": 30, "desc": "20-30 маток на верблюда"},
    "хряка": {"min_females": 50, "max_females": 200, "desc": "50-200 свиноматок на хряка"},
}

# ========== МИНИМАЛЬНОЕ ПОГОЛОВЬЕ ДЛЯ КОНКРЕТНЫХ СУБСИДИЙ ==========
MIN_HERD_RULES = {
    "молока (коровье) с фуражным поголовьем коров от 600": 600,
    "молока (коровье) с фуражным поголовьем коров от 400": 400,
    "молока (коровье) с фуражным поголовьем коров от 50": 50,
    "свиней, реализованных": 200,  # мин 200 свиноматок
    "молока (кобылье)": 30,  # мин 30 голов маточных
    "молока (верблюжье)": 30,  # мин 30 голов маточных
}

# ========== ЛИМИТЫ ПРОИЗВОДСТВА ДЛЯ ПТИЦЕВОДСТВА ==========
POULTRY_PRODUCTION = {
    80: 15000,  # от 15000 тонн
    70: 10000,  # от 10000 тонн
    60: 5000,   # от 5000 тонн
    50: 500,    # от 500 тонн
}

# ========== НОРМЫ ПАДЕЖА (из нового PDF) ==========
MORTALITY_NORMS = {
    "cattle_meat_mature": 0.02,       # маточное мясное — 2%
    "cattle_meat_calves": 0.02,       # телята мясные до отъёма — 2%
    "cattle_meat_young": 0.02,        # молодняк на откорме — 2%
    "cattle_milk_mature": 0.03,       # маточное молочное — 3%
    "cattle_milk_calves": 0.035,      # телята молочные — 3.5%
    "cattle_import_auto": 0.05,       # импорт авто >2000км — 5%
    "cattle_import_sea": 0.075,       # импорт морской — 7.5%
    "sheep_mature": 0.03,             # взрослые овцы — 3%
    "sheep_lambs": 0.05,              # ягнята до отъёма — 5%
    "sheep_young": 0.02,              # молодняк — 2%
    "horse_foals": 0.023,             # жеребята — 2.3%
    "horse_young": 0.027,             # молодняк — 2.7%
    "camel_calves": 0.06,             # верблюжата — 6%
    "pig_piglets": 0.125,             # поросята-сосуны — 12.5%
    "pig_growers": 0.052,             # поросята на доращивании — 5.2%
    "pig_fattening": 0.01,            # молодняк на откорме — 1%
    "poultry_meat": 0.075,            # мясная птица — 7.5%
    "poultry_egg": 0.075,             # яичная птица — 7.5%
    "beekeeping": 0.20,               # пчелосемьи — 20%
}

# ========== НОРМЫ ПАСТБИЩ (средние по областям, га на 1 КРС) ==========
PASTURE_NORMS = {
    "Абай": 10.0,
    "Акмолин": 8.5,
    "Актюбин": 11.0,
    "Алматин": 14.0,
    "Атырау": 18.0,
    "Восточно": 9.0,
    "Жамбыл": 10.0,
    "Западно": 10.0,
    "Караганд": 12.0,
    "Костанай": 8.0,
    "Кызылорд": 16.0,
    "Мангистау": 14.0,
    "Павлодар": 9.0,
    "Северо": 5.0,
    "Туркестан": 10.0,
    "Ұлытау": 12.0,
    "Жетісу": 14.0,
    "Шымкент": 12.0,
}

# ========== СРОКИ ПОДАЧИ ==========
SUBMISSION_WINDOW = {
    "default": (1, 20, 12, 20),  # 20 января — 20 декабря
    "осеменению маточного поголовья овец": (9, 1, 12, 20),  # 1 сентября — 20 декабря
}

# ========== ПРАВИЛО 50% ==========
# Субсидия не более 50% стоимости приобретения (для племенных, кроме птиц и свиней)
FIFTY_PERCENT_EXEMPT = {"suточного молодняка", "свиней", "свинок", "хряка"}


def check_rules(df):
    """Проверяет каждую заявку. Возвращает df с rule_* колонками и comp_rules."""
    df = df.copy()
    n = len(df)
    sub_name = df.get("subsidy_name", pd.Series([""] * n, index=df.index)).fillna("")
    direction = df.get("direction_code", pd.Series(["other"] * n, index=df.index))
    rate = df["rate"]
    amount = df["amount"]

    # ---- 1. Дата подачи: 20 января — 20 декабря ----
    if "submitted_at" in df.columns:
        m = df["submitted_at"].dt.month
        d = df["submitted_at"].dt.day
        too_early = (m == 1) & (d < 20)
        too_late = (m == 12) & (d > 20)
        df["rule_date_window"] = (~too_early & ~too_late).astype(int)
    else:
        df["rule_date_window"] = 1

    # ---- 2. Норматив из списка допустимых ----
    def rate_ok(dir_code, r):
        valid = VALID_RATES_BY_DIRECTION.get(dir_code, set())
        return 1 if (not valid or r in valid) else 0
    df["rule_valid_rate"] = [rate_ok(d, r) for d, r in zip(direction, rate)]

    # ---- 3. Количество единиц > 0 ----
    if "unit_count" in df.columns:
        df["rule_positive_units"] = (df["unit_count"] > 0).astype(int)
    else:
        df["rule_positive_units"] = 1

    # ---- 4. Разумный объём по направлению ----
    if "unit_count" in df.columns:
        max_units = direction.map({
            "poultry": 10_000_000, "beekeeping": 100_000,
        }).fillna(50_000)
        df["rule_reasonable_units"] = (df["unit_count"] <= max_units).astype(int)
    else:
        df["rule_reasonable_units"] = 1

    # ---- 5. Сумма > 0 ----
    df["rule_positive_amount"] = (amount > 0).astype(int)

    # ---- 6. Рабочие часы (7:00+) ----
    if "submit_hour" in df.columns:
        df["rule_business_hours"] = (df["submit_hour"] >= 7).astype(int)
    else:
        df["rule_business_hours"] = 1

    # ---- 7. Минимальное поголовье для молочных/свиных субсидий ----
    if "unit_count" in df.columns:
        min_check = np.ones(n, dtype=int)
        for keyword, min_heads in MIN_HERD_RULES.items():
            import re
            escaped = re.escape(keyword)
            mask = sub_name.str.contains(escaped, case=False, na=False)
            min_check[mask & (df["unit_count"].values < min_heads)] = 0
        df["rule_min_herd"] = min_check
    else:
        df["rule_min_herd"] = 1

    # ---- 8. Норматив птицеводства vs объём производства ----
    poultry_check = np.ones(n, dtype=int)
    if "unit_count" in df.columns:
        for norm_rate, min_tons in POULTRY_PRODUCTION.items():
            mask = (direction == "poultry") & (rate == norm_rate)
            # unit_count в кг, min_tons в тоннах -> конвертируем
            kg_threshold = min_tons * 1000
            poultry_check[mask & (df["unit_count"].values < kg_threshold)] = 0
    df["rule_poultry_volume"] = poultry_check

    # ---- 9. Правило 50%: субсидия ≤ 50% стоимости ----
    # Proxy: если сумма > 2x нормального для данного типа = подозрительно
    fifty_check = np.ones(n, dtype=int)
    for keyword in FIFTY_PERCENT_EXEMPT:
        pass  # Освобождены от правила
    # Для остальных племенных: сумма на единицу не должна быть > норматива
    # (норматив УЖЕ ≤ 50%, так что если сумма/единицы > норматива = нарушение)
    if "unit_count" in df.columns:
        actual_per_unit = amount / df["unit_count"].replace(0, np.nan)
        over_rate = actual_per_unit > rate * 1.05  # 5% допуск на округление
        # Только для приобретения (не для удешевления и работы)
        is_acquisition = sub_name.str.contains("приобретение", case=False, na=False)
        fifty_check[over_rate & is_acquisition] = 0
    df["rule_50pct_limit"] = fifty_check

    # ---- 10. Региональная нагрузка пастбищ ----
    # Если unit_count слишком велик для района — флаг
    pasture_check = np.ones(n, dtype=int)
    if "unit_count" in df.columns and "oblast" in df.columns:
        for oblast_key, ha_per_head in PASTURE_NORMS.items():
            mask = df["oblast"].str.contains(oblast_key, case=False, na=False)
            mask = mask & direction.isin(["cattle", "horse", "camel", "sheep"])
            if ha_per_head >= 14:
                pasture_check[mask & (df["unit_count"].values > 5000)] = 0
            elif ha_per_head >= 10:
                pasture_check[mask & (df["unit_count"].values > 10000)] = 0
    df["rule_pasture_capacity"] = pasture_check

    # ---- 11. Сезонность: МРС сезонные поставки только январь-апрель ----
    seasonal_check = np.ones(n, dtype=int)
    if "submitted_at" in df.columns:
        is_seasonal = sub_name.str.contains("сезонные поставки", case=False, na=False)
        month = df["submitted_at"].dt.month
        seasonal_check[is_seasonal & (month > 4)] = 0  # подано после апреля
    df["rule_seasonal_window"] = seasonal_check

    # ========== СОБИРАЕМ БАЛЛ ==========
    rule_cols = [c for c in df.columns if c.startswith("rule_")]

    rule_weights = {
        "rule_date_window": 10,
        "rule_valid_rate": 20,
        "rule_positive_units": 10,
        "rule_reasonable_units": 10,
        "rule_positive_amount": 5,
        "rule_business_hours": 5,
        "rule_min_herd": 15,
        "rule_poultry_volume": 10,
        "rule_50pct_limit": 10,
        "rule_pasture_capacity": 10,
        "rule_seasonal_window": 5,
    }

    total_w = sum(rule_weights.get(c, 5) for c in rule_cols)
    df["comp_rules"] = sum(
        df[col] * rule_weights.get(col, 5) / total_w * 100
        for col in rule_cols
    ).round(1)

    # Список нарушений
    violations = []
    for idx in df.index:
        v = []
        for col in rule_cols:
            if df.loc[idx, col] == 0:
                v.append(col.replace("rule_", "").replace("_", " "))
        violations.append(v)
    df["rule_violations"] = violations
    df["rule_violation_count"] = df["rule_violations"].apply(len)

    return df


# ========== Функции для других модулей ==========

def get_mortality_norm(direction, subsidy_name=""):
    """Возвращает норму падежа для данного типа."""
    sn = subsidy_name.lower()
    if direction == "cattle":
        if "молоч" in sn: return MORTALITY_NORMS["cattle_milk_mature"]
        if "импорт" in sn: return MORTALITY_NORMS["cattle_import_auto"]
        return MORTALITY_NORMS["cattle_meat_mature"]
    elif direction == "sheep": return MORTALITY_NORMS["sheep_mature"]
    elif direction == "horse": return MORTALITY_NORMS["horse_foals"]
    elif direction == "camel": return MORTALITY_NORMS["camel_calves"]
    elif direction == "pig": return MORTALITY_NORMS["pig_fattening"]
    elif direction == "poultry": return MORTALITY_NORMS["poultry_meat"]
    elif direction == "beekeeping": return MORTALITY_NORMS["beekeeping"]
    return 0.03  # default

def get_pasture_norm(oblast):
    """Возвращает норму га/голова для области."""
    oblast_str = str(oblast).lower()
    for key, val in PASTURE_NORMS.items():
        if key.lower() in oblast_str:
            return val
    return 10.0  # default


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_and_prepare

    df = load_and_prepare()
    df = check_rules(df)

    print("=== RULE CHECK RESULTS (EXPANDED) ===")
    rule_cols = [c for c in df.columns if c.startswith("rule_") and c not in ["rule_violations","rule_violation_count"]]
    for col in sorted(rule_cols):
        passed = df[col].sum()
        total = len(df)
        print(f"  {col:30s} passed: {passed:>6}/{total} ({passed/total:.1%})")

    print(f"\nComp_rules distribution:")
    for pct in [100, 95, 90, 85, 80, 75, 50]:
        cnt = (df["comp_rules"] == pct).sum() + (df["comp_rules"].between(pct-2, pct+2)).sum()
    print(df["comp_rules"].describe())

    print(f"\nViolation count:")
    print(df["rule_violation_count"].value_counts().sort_index())

    # Корреляция с реальными решениями
    for status in ["Исполнена", "Отклонена"]:
        subset = df[df["status"] == status]
        avg = subset["comp_rules"].mean()
        viols = subset["rule_violation_count"].mean()
        print(f"\n{status:15s}: avg score={avg:.1f}, avg violations={viols:.2f}")

    # Какие правила чаще нарушаются у отклонённых?
    print("\nViolation breakdown for rejected applications:")
    rejected = df[df["status"]=="Отклонена"]
    for col in sorted(rule_cols):
        fail_rate = 1 - rejected[col].mean()
        if fail_rate > 0.001:
            print(f"  {col:30s} fail rate: {fail_rate:.1%}")
