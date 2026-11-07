import importlib
import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")

@pytest.fixture
def toy():
    """Dataset sintético con las columnas candidatas, para probar sin depender de Silver."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n)),
        "SibSp": rng.integers(0, 4, n),
        "Parch": rng.integers(0, 3, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 6, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(list("XXXXCB"), n),
    })
    y = pd.Series(((X["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int))
    return X, y

def test_group_age_imputer_uses_only_training_statistics():
    train = pd.DataFrame({"Sex": ["male", "male", "female"], "Age": [10.0, 20.0, 40.0], "Pclass": [1, 1, 1]})
    test = pd.DataFrame({"Sex": ["male", "other"], "Age": [np.nan, np.nan], "Pclass": [1, 1]})
    out = ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert out.loc[0, "Age"] == 15.0                          # promedio de hombres en TRAIN
    assert out.loc[1, "Age"] == pytest.approx(70 / 3)         # grupo no visto: promedio global de TRAIN

def test_pipeline_fits_and_predicts_probabilities(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((0 <= proba) & (proba <= 1)).all()

def test_split_is_stratified_and_reproducible(toy):
    X, y = toy
    a = ml.split(X, y, test_size=0.25, random_state=1)
    b = ml.split(X, y, test_size=0.25, random_state=1)
    pd.testing.assert_index_equal(a[0].index, b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03

def test_candidates_include_baseline_and_are_seeded():
    cands = ml.get_candidates(42)
    assert "dummy" in cands and "logreg" in cands and len(cands) >= 5

def test_evaluate_on_test_returns_metrics(toy):
    X, y = toy
    X_tr, X_te, y_tr, y_te = ml.split(X, y, test_size=0.25, random_state=1)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    res = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.5)
    assert {"accuracy", "f1", "roc_auc", "n_test"} <= set(res)
    assert 0 <= res["accuracy"] <= 1 and res["n_test"] == len(X_te)