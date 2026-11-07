"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame: ...      # contrato de entrada
def cast_types(df: pd.DataFrame) -> pd.DataFrame: ...           # Survived -> int, etc.
def add_person(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 8
def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame: ...   # Pasos 11 y 24 (TCabin, HasCabin)
def add_family_features(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 14 (Familia, Alone)
def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 16 (Survivor)
def add_title(df: pd.DataFrame) -> pd.DataFrame: ...            # Paso 25 (TTitulo)
def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 26
def impute_embarked(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 21
def impute_age(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 22
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame: ...  # Pasos 21, 23, 24, 26
def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 32
def validate_silver(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 33


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME