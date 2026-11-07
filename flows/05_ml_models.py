"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta."""
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None):
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    ts = test_size if test_size is not None else getattr(settings, "TEST_SIZE", 0.2)
    rs = random_state if random_state is not None else getattr(settings, "RANDOM_STATE", 42)
    
    return train_test_split(
        X, y,
        test_size=ts,
        random_state=rs,
        stratify=y
    )


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Imputador de edad basado en la media de grupos específicos (ej. Sex, Pclass)."""
    def __init__(self, group_cols=('Sex', 'Pclass')):
        self.group_cols = group_cols
        self.means_ = {}
        self.global_mean_ = 0.0

    def fit(self, X, y=None):
        X_ = X.copy()
        self.global_mean_ = X_['Age'].mean()
        cols = list(self.group_cols)
        self.means_ = X_.groupby(cols)['Age'].mean().to_dict()
        return self

    def transform(self, X):
        X_ = X.copy()
        if 'Age' not in X_.columns:
            return X_
        
        cols = list(self.group_cols)
        def impute_row(row):
            if pd.isna(row['Age']):
                if len(cols) == 1:
                    val = row[cols[0]]
                    if val in self.means_:
                        return self.means_[val]
                    elif (val,) in self.means_:
                        return self.means_[(val,)]
                    return self.global_mean_
                else:
                    key = tuple(row[col] for col in cols)
                    return self.means_.get(key, self.global_mean_)
            return row['Age']
            
        X_['Age'] = X_.apply(impute_row, axis=1)
        return X_


def build_preprocessor(numeric_cols: list | None = None, categorical_cols: list | None = None) -> ColumnTransformer:
    """Construye un ColumnTransformer para procesar variables numéricas y categóricas por separado."""
    num_cols = numeric_cols if numeric_cols is not None else NUM_FEATURES
    cat_cols = categorical_cols if categorical_cols is not None else CAT_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        sparse_threshold=0
    )
    
    return preprocessor


def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline."""
    prep = build_preprocessor(num_features, cat_features)
    
    return Pipeline(steps=[
        ('prep', prep),
        ('model', model)
    ])


def get_candidates(random_state: int = 42) -> dict:
    """Diccionario {nombre: estimador sin entrenar} con una familia variada de modelos."""
    return {
        "dummy": DummyClassifier(strategy='most_frequent'),
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }

def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None) -> pd.DataFrame:
    """Devuelve un DataFrame con media y desviación de cada métrica por modelo."""
    rs = random_state if random_state is not None else getattr(settings, "RANDOM_STATE", 42)
    folds = cv_folds if cv_folds is not None else 5
    
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    candidates = get_candidates(rs)
    
    records = []
    for name, model in candidates.items():
        pipe = build_pipeline(model, num_features, cat_features)
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
        
        record = {"model": name}
        for metric in SCORING:
            test_scores = res[f"test_{metric}"]
            train_scores = res[f"train_{metric}"]
            record[f"test_{metric}_mean"] = test_scores.mean()
            record[f"test_{metric}_std"] = test_scores.std()
            record[f"train_{metric}_mean"] = train_scores.mean()
            record[f"train_{metric}_std"] = train_scores.std()
            
        records.append(record)
        
    df_results = pd.DataFrame(records)
    sort_col = f"test_{PRIMARY_METRIC}_mean"
    if sort_col in df_results.columns:
        df_results = df_results.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        
    return df_results

def get_param_grids() -> dict[str, dict]:
    """Rejillas por modelo, con los nombres que espera el Pipeline (prefijo 'model__')."""
    return {
        'logreg': {
            'model__C': [0.01, 0.1, 1, 10], 
            'model__penalty': ['l2']
        },
        'forest': {
            'model__n_estimators': [200, 400], 
            'model__max_depth': [4, 6, 8, None],
            'model__min_samples_leaf': [1, 3, 5]
        },
        'hgb': {
            'model__learning_rate': [0.03, 0.1], 
            'model__max_depth': [2, 3, 4],
            'model__max_iter': [100, 200], 
            'model__l2_regularization': [0.0, 1.0]
        }
    }

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba (dict serializable a JSON)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    try:
        roc_auc = roc_auc_score(y_test, proba)
    except ValueError:
        roc_auc = float('nan')
        
    return {
        "accuracy": float(acc),
        "f1": float(f1),             # <-- Cambiado de f1_score a f1
        "roc_auc": float(roc_auc),
        "n_test": int(len(y_test))   # <-- Cambiado de n_samples a n_test
    }