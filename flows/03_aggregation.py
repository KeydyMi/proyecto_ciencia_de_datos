"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.pyplot import table
import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:             # Paso 28
    """Genera cuatro tablas estadísticas agrupadas con groupby y devuelve un diccionario."""
    age_by_survival = df.groupby('Survived')['Age'].describe().reset_index()
    survival_by_sex = df.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    survival_by_pclass = df.groupby('Pclass')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    survival_by_alone = df.groupby('Alone')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    return {
        'age_by_survival': age_by_survival,
        'survival_by_sex': survival_by_sex,
        'survival_by_pclass': survival_by_pclass,
        'survival_by_alone': survival_by_alone
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:                          # Paso 29
    """Calcula la prueba de independencia Chi-cuadrado entre 'Sex' y 'Survived'."""
    tabla = pd.crosstab(df['Sex'], df['Survived'])
    chi2, p, dof, esperado = stats.chi2_contingency(tabla)
    return {
        'chi2': float(chi2),
        'p_value': float(p),
        'dof': int(dof)
    }


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:              # Paso 34
   """Supervivencia cruzada por Clase y Sexo (tasa de supervivencia)."""
   tabla: pd.DataFrame = pd.pivot_table(
        df, 
        values='Survived', 
        index='Pclass', 
        columns='Sex', 
        aggfunc='mean'
    )
   return tabla.reset_index().rename(columns=str)


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:               # Paso 35 (1)
    """Supervivencia por puerto de embarque."""
    return df.groupby('Embarked')['Survived'].agg(['mean', 'sum', 'count']).reset_index().rename(columns=str)


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:         # Paso 35 (2)
    """Supervivencia estratificada por Puerto y Clase (variable de confusión)."""
    tabla = pd.pivot_table(
        df, 
        values='Survived', 
        index='Embarked', 
        columns='Pclass', 
        aggfunc='mean'
    )
    return tabla.reset_index().rename(columns=str)


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:                  # Paso 36 (1)
    """Añade cuartiles de tarifa (Fare) calculados por clase con etiquetas enteras."""
    df = df.copy()
    df['FareQ'] = df.groupby('Pclass')['Fare'].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates='drop')
    )
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:    # Paso 36 (2)
    """Supervivencia por cuartil de tarifa y clase."""
    df_temp = add_fare_quartile(df)
    tabla = pd.pivot_table(
        df_temp, 
        values='Survived', 
        index='FareQ', 
        columns='Pclass', 
        aggfunc='mean'
    )
    return tabla.reset_index().rename(columns=str)


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:                 # Paso 37
    """Supervivencia según el tamaño de la familia (SibSp + Parch + 1)."""
    if 'Familia' not in df.columns:
        df = df.copy()
        df['Familia'] = df['SibSp'] + df['Parch'] + 1
    return df.groupby('Familia')['Survived'].agg(['mean', 'sum', 'count']).reset_index().rename(columns=str)


def outlier_mask_iqr(s: pd.Series) -> pd.Series:                          # Paso 38 (1)
    """Máscara booleana para detectar atípicos usando el rango intercuartílico (IQR)."""
    q = s.quantile([0.25, 0.75])
    Q1 = q.loc[0.25]
    Q3 = q.loc[0.75]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (s < lower_bound) | (s > upper_bound)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:                    # Paso 38 (2-3)
    """Resumen de valores atípicos detectados en columnas numéricas clave."""
    # Verificamos si existe AgeOriginal, de lo contrario usamos Age
    age_col = 'AgeOriginal' if 'AgeOriginal' in df.columns else 'Age'
    cols_to_check = ['Fare', age_col, 'Familia']
    
    summary_data = []
    for col in cols_to_check:
        if col in df.columns:
            mask = outlier_mask_iqr(df[col])
            summary_data.append({
                'variable': col,
                'outliers_count': int(mask.sum()),
                'total_count': int(len(df)),
                'pct_outliers': float(mask.mean() * 100)
            })
            
    return pd.DataFrame(summary_data).rename(columns=str)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir