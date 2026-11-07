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
# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"                 # Nombre del mejor modelo en get_candidates()
FINAL_PARAMS = {                        # Mejores hiperparámetros obtenidos en el GridSearch (Paso 57)
    "model__learning_rate": 0.05,
    "model__max_depth": 3,
    "model__max_iter": 200,
    "model__l2_regularization": 1.0,
}
USE_CABIN_FEATURES = False              # Decisión de ablación del Paso 61 (sin cabinas)
THRESHOLD = 0.30                        # Umbral óptimo de decisión (Paso 53)

from sklearn.pipeline import Pipeline
import pandas as pd

def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Construye el pipeline con las decisiones congeladas y lo entrena con TODO (X, y)."""
    # 1. Definir columnas numéricas y categóricas según si se usan o no las características de cabina
    if USE_CABIN_FEATURES:
        num = NUM_FEATURES + CABIN_NUM
        cat = CAT_FEATURES + CABIN_CAT
    else:
        num = NUM_FEATURES
        cat = CAT_FEATURES

    # 2. Obtener el modelo candidato seleccionado congelado
    modelo = get_candidates(42)[FINAL_ESTIMATOR]

    # 3. Construir el pipeline pasando las listas de columnas y aplicando los hiperparámetros congelados
    pipe = build_pipeline(modelo, num, cat).set_params(**FINAL_PARAMS)

    # 4. Ajustar y retornar el pipeline entrenado con TODOS los datos disponibles
    return pipe.fit(X, y)
import joblib
import json
import platform
import sklearn
from datetime import datetime, timezone

def save_model(pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME,
               models_dir: Path = MODELS) -> Path:
    """Guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.pandas.__version__ if hasattr(pd, 'pandas') else pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path

def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Modelo entrenado con scikit-learn %s; versión actual %s",
                    meta.get("sklearn_version"), sklearn.__version__)
    return pipeline, meta

def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Silver -> evaluación honesta -> modelo final entrenado y guardado."""
    log.info("Iniciando flujo completo de modelado (Paso 68)...")
    
    # 1. Cargar el dataset completo desde Silver
    data_path = Path(in_dir) / SILVER_NAME if in_dir else (SILVER / SILVER_NAME)
    X, y = load_dataset(data_path)
    
    # 2. Dividir para la evaluación honesta inicial
    X_tr, X_te, y_tr, y_te = split(X, y)
    
    # 3. Evaluación honesta: entrenar con X_tr y evaluar en X_te con el umbral congelado
    # Construimos un pipeline temporal solo para auditar el rendimiento esperado
    num = NUM_FEATURES + CABIN_NUM if USE_CABIN_FEATURES else NUM_FEATURES
    cat = CAT_FEATURES + CABIN_CAT if USE_CABIN_FEATURES else CAT_FEATURES
    modelo_eval = get_candidates(42)[FINAL_ESTIMATOR]
    pipe_eval = build_pipeline(modelo_eval, num, cat).set_params(**FINAL_PARAMS)
    pipe_eval.fit(X_tr, y_tr)
    
    metrics = evaluate_on_test(pipe_eval, X_te, y_te, THRESHOLD)
    log.info("Métricas de evaluación honesta en prueba: %s", metrics)
    
    # 4. Modelo definitivo: entrenar con TODOS los datos etiquetados (X, y)
    final_pipeline = train_final(X, y)
    
    # 5. Preparar metadatos completos exigidos
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "features": {
            "num": num,
            "cat": cat
        },
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": getattr(settings, "RANDOM_STATE", 42),
        "n_train": int(len(X)),
        "data_sha256": sha256_of(data_path),
        "metrics": metrics
    }
    
    # 6. Guardar el modelo definitivo y sus metadatos
    model_path = save_model(final_pipeline, metadata, models_dir=Path(models_dir))
    log.info("¡Flujo completado con éxito! Modelo guardado en %s", model_path)
    
    return model_path

def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Devuelve un DataFrame con 'proba_survive' y 'pred' (0/1) para cada fila de X."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    result_df = pd.DataFrame({
        "proba_survive": proba,
        "pred": pred
    }, index=X.index)
    
    return result_df
def test_save_and_load_roundtrip(tmp_path, toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    class ml:
        ...

    path = ml.save_model(pipe, {"estimator": "logreg", "n_train": len(X)},
                         name="toy_model", models_dir=tmp_path)
    loaded, meta = ml.load_model(path)
    assert np.array_equal(pipe.predict(X), loaded.predict(X))     # mismas predicciones
    assert {"sklearn_version", "python_version", "model_sha256", "saved_at"} <= set(meta)
    assert meta["n_train"] == len(X)

def test_predict_returns_probabilities_and_labels(toy):
    X, y = toy
    class ml:
        ...

    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    out = ml.predict(pipe, X, threshold=0.5)
    assert list(out.columns) == ["proba_survive", "pred"]
    assert set(out["pred"].unique()) <= {0, 1}