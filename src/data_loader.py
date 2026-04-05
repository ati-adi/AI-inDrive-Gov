"""
data_loader.py — читает Excel с заявками, чистит данные, выводит базовые фичи.

Вход:  data/raw/Выгрузка_по_выданным_субсидиям_2025_год_обезлич.xlsx
Выход: pd.DataFrame с чистыми колонками + derived фичи
"""

import pandas as pd
import numpy as np
from pathlib import Path


COLUMN_MAP = {
    "№ п/п": "app_index",
    "Дата поступления": "submitted_at",
    "Область": "oblast",
    "Акимат": "akimat",
    "Номер заявки": "app_number",
    "Направление водства": "subsidy_direction",
    "Наименование субсидирования": "subsidy_name",
    "Статус заявки": "status",
    "Норматив": "rate",
    "Причитающая сумма": "amount",
    "Район хозяйства": "district",
}

STATUS_LABELS = {
    "Исполнена": 1,
    "Одобрена": 1,
    "Сформировано поручение": 1,
    "Отклонена": 0,
    "Отозвано": -1,   # exclude from training
    "Получена": -1,   # exclude from training
}

DIRECTION_SHORT = {
    "Субсидирование в скотоводстве": "cattle",
    "Субсидирование в птицеводстве": "poultry",
    "Субсидирование в овцеводстве": "sheep",
    "Субсидирование в верблюдоводстве": "camel",
    "Субсидирование в коневодстве": "horse",
    "Субсидирование затрат по искусственному осеменению": "insemination",
    "Субсидирование в пчеловодстве": "beekeeping",
    "Субсидирование в свиноводстве": "pig",
    "Субсидирование в козоводстве": "goat",
}


def load_raw(path=None):
    if path is None:
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        upload = raw_dir / "upload.xlsx"
        original = raw_dir / "Выгрузка_по_выданным_субсидиям_2025_год_обезлич.xlsx"
        if upload.exists():
            path = upload
        elif original.exists():
            path = original
        else:
            xlsx_files = list(raw_dir.glob("*.xlsx"))
            if xlsx_files:
                path = xlsx_files[0]
            else:
                raise FileNotFoundError(f"No .xlsx files in {raw_dir}")
    return pd.read_excel(path, header=4)


def clean(df):
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    df = df.rename(columns=COLUMN_MAP)
    df = df.dropna(subset=["app_number", "status"], how="any")
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["rate", "amount"])
    for col in ["oblast", "district", "subsidy_direction", "subsidy_name", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


def add_derived_features(df):
    # Количество голов = сумма / норматив
    df["unit_count"] = (df["amount"] / df["rate"]).round(1)

    # Временные фичи
    df["submit_hour"] = df["submitted_at"].dt.hour
    df["submit_dow"] = df["submitted_at"].dt.dayofweek
    df["submit_month"] = df["submitted_at"].dt.month

    # Короткий код направления
    df["direction_code"] = df["subsidy_direction"].map(DIRECTION_SHORT).fillna("other")

    # ML label
    df["label"] = df["status"].map(STATUS_LABELS)

    # Региональные статистики
    df["amount_vs_district"] = df["amount"] / df.groupby("district")["amount"].transform("mean")
    df["units_vs_district"] = df["unit_count"] / df.groupby("district")["unit_count"].transform("mean")
    df["district_app_count"] = df.groupby("district")["app_number"].transform("count")

    approved = df[df["label"] == 1].groupby("district").size()
    total = df[df["label"].isin([0, 1])].groupby("district").size()
    dist_rate = (approved / total).fillna(0)
    df["district_approve_rate"] = df["district"].map(dist_rate).fillna(0)

    return df


def load_and_prepare(path=None):
    """Полный пайплайн: загрузка -> чистка -> derived features."""
    df = load_raw(path)
    df = clean(df)
    df = add_derived_features(df)
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    print(f"Loaded {len(df)} applications")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nSample:")
    print(df[["district","direction_code","unit_count","amount_vs_district","submit_hour","label"]].head(10))
