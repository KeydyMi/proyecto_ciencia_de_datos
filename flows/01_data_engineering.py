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

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
]

def validate_bronze(df: pd.DataFrame) -> pd.DataFrame: 
    """Valida el contrato de Bronze: columnas requeridas y sin PassengerId duplicados."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en Bronze: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Existen valores duplicados en 'PassengerId'")
    return df      # contrato de entrada

def cast_types(df: pd.DataFrame) -> pd.DataFrame: 
    """Convierte los tipos de datos a entero o flotante según corresponda."""
    df = df.copy()
    int_cols = ['Survived', 'Pclass', 'SibSp', 'Parch']
    float_cols = ['Age', 'Fare']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df          # Survived -> int, etc.

def add_person(df: pd.DataFrame) -> pd.DataFrame: 
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    df = df.copy()
    df['Person'] = np.where(df['Age'] < 16, 'child', df['Sex'])
    return df         # Paso 8

def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
     """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
     df = df.copy()
     df['TCabin'] = df['Cabin'].str[0].fillna('X')
     df['HasCabin'] = df['Cabin'].notna().astype(int) 
     return df # Pasos 11 y 24 (TCabin, HasCabin)

def add_family_features(df: pd.DataFrame) -> pd.DataFrame: 
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    df = df.copy()
    df['Familia'] = df['SibSp'] + df['Parch']
    df['Alone'] = np.where(df['Familia'] > 0, 'with Family', 'Alone')
    return df # Paso 14 (Familia, Alone)

def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame: 
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    df = df.copy()
    df['Survivor'] = df['Survived'].map({0: 'No', 1: 'Yes'})
    return df   # Paso 16 (Survivor)

def add_title(df: pd.DataFrame) -> pd.DataFrame: 
    """Extrae el título del nombre del pasajero usando expresiones regulares y crea 'TTitulo'."""
    df = df.copy()
    # Extrae el texto entre la coma y el punto, y elimina espacios sobrantes con .str.strip()
    df['TTitulo'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    return df         # Paso 25 (TTitulo)

# Constantes para el manejo de títulos (Paso 26)
TITLE_MAP = {
    'Master': 'Mr',
    'Mme': 'Mrs',
    'Dona': 'Mrs',
    'the Countess': 'Mrs',
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Lady': 'Miss'
}
MAIN_TITLES = ['Mr', 'Mrs', 'Miss']

def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame: 
    """Consolida títulos raros usando TITLE_MAP y agrupa los restantes en 'Otro'."""
    df = df.copy()
    # 1. Reemplazamos títulos usando el diccionario TITLE_MAP
    df['TTitulo'] = df['TTitulo'].replace(TITLE_MAP)
    # 2. Todo título que no esté en MAIN_TITLES pasa a 'Otro'
    df['TTitulo'] = df['TTitulo'].where(df['TTitulo'].isin(MAIN_TITLES), 'Otro')
    return df   # Paso 26

def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los valores faltantes en 'Embarked' con la categoría más frecuente (moda)."""
    df = df.copy()
    moda_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(moda_embarked)
    return df     # Paso 21

def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    df = df.copy()
    
    # 1. Guardamos la edad original y creamos la marca booleana de imputación
    df['AgeOriginal'] = df['Age']
    df['AgeImputada'] = df['Age'].isna()
    
    # 2. Calculamos los promedios de edad agrupados por 'Sex'
    promedios = df.groupby('Sex')['Age'].mean()
    
    # 3. Rellenamos los nulos de 'Age' mapeando con los promedios calculados por género
    df['Age'] = df['Age'].fillna(df['Sex'].map(promedios))
    
    return df          # Paso 22

# Definimos la constante DECKS fija para NCabin (Paso 24)
DECKS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X']

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas codificadas numéricamente como 'NEmbarked', 'NSex', 'NCabin' y 'NTitulo'."""
    df = df.copy()
    
    df['NEmbarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['NSex'] = df['Sex'].map({'female': 0, 'male': 1})
    
    deck_map = {letra: i for i, letra in enumerate(DECKS)}
    df['NCabin'] = df['TCabin'].map(deck_map)
    
    # 3. NTitulo (Paso 26): Mr -> 0, Mrs -> 1, Miss -> 2, Otro -> 3
    df['NTitulo'] = df['TTitulo'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Otro': 3})
    
    return df # Pasos 21, 23, 24, 26

def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin las columnas originales Name, Sex, Embarked y Cabin."""
    df = df.copy()
    # Eliminamos las columnas de texto originales que ya tienen sus equivalentes numéricos/derivados
    cols_to_drop = ['Name', 'Sex', 'Embarked', 'Cabin']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df  # Paso 32

def validate_silver(df: pd.DataFrame) -> pd.DataFrame: 
    """Valida el contrato de Silver: sin nulos en columnas clave (AgeOriginal sí puede tenerlos)."""
    key_cols = [
        'Survived', 'Pclass', 'Age', 'Fare', 'Embarked', 'Person', 
        'Alone', 'TCabin', 'TTitulo', 'NSex', 'NEmbarked', 'NCabin', 'NTitulo'
    ]
    for col in key_cols:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"La columna clave '{col}' contiene valores nulos en la capa Silver.")
    return df      # Paso 33


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