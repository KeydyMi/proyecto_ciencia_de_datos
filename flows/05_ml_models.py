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