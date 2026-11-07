import importlib

import numpy as np
import pandas as pd
import pytest

de = importlib.import_module("flows.01_data_engineering")


@pytest.fixture
def sample():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
            "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
            "Moran, Dr. James",
        ],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22.0, 38.0, np.nan, 2.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 3, 0, 0],
        "Parch": [0, 0, 0, 1, 0, 0],
        "Ticket": list("ABCDEF"),
        "Fare": [7.25, 71.28, 7.92, 21.07, 86.5, 8.46],
        "Cabin": [None, "C85", None, None, "B77", None],
        "Embarked": ["S", "C", "S", None, "S", "Q"],
    })


def test_validate_bronze_rejects_missing_columns(sample):
    with pytest.raises(ValueError):
        de.validate_bronze(sample.drop(columns=["Fare"]))


def test_add_person_is_explicit_about_unknown_age(sample):
    out = de.add_person(sample)
    assert out["Person"].tolist() == ["male", "female", "female", "child", "female", "male"]


def test_family_features(sample):
    out = de.add_family_features(sample)
    assert out["Familia"].tolist() == [1, 1, 0, 4, 0, 0]
    assert out["Alone"].tolist() == ["with Family", "with Family", "Alone",
                                     "with Family", "Alone", "Alone"]


def test_cabin_features(sample):
    out = de.add_cabin_features(sample)
    assert out["TCabin"].tolist() == ["X", "C", "X", "X", "B", "X"]
    assert out["HasCabin"].tolist() == [0, 1, 0, 0, 1, 0]


def test_titles_are_extracted_and_consolidated(sample):
    out = de.consolidate_titles(de.add_title(sample))
    assert out["TTitulo"].tolist() == ["Mr", "Mrs", "Miss", "Mr", "Mrs", "Otro"]


def test_impute_embarked_uses_mode(sample):
    out = de.impute_embarked(sample)
    assert out["Embarked"].isna().sum() == 0
    assert out.loc[3, "Embarked"] == "S"


def test_impute_age_by_sex_keeps_original(sample):
    out = de.impute_age(sample)
    assert out["Age"].isna().sum() == 0
    assert out.loc[2, "Age"] == pytest.approx(35.5)     # promedio de mujeres: (38 + 33) / 2
    assert out.loc[5, "Age"] == pytest.approx(12.0)     # promedio de hombres: (22 + 2) / 2
    assert out["AgeOriginal"].isna().sum() == 2
    assert out["AgeImputada"].tolist() == [False, False, True, False, False, True]


def test_functions_do_not_mutate_their_input(sample):
    before = sample.copy(deep=True)
    for fn in (de.add_person, de.add_family_features, de.add_cabin_features,
               de.impute_embarked, de.impute_age):
        fn(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_build_silver_end_to_end(sample):
    out = de.validate_silver(de.build_silver(sample))
    assert len(out) == len(sample)
    for col in ["Person", "Alone", "TCabin", "TTitulo", "NSex", "NEmbarked", "NCabin",
                "NTitulo", "AgeOriginal", "Survivor"]:
        assert col in out.columns
    assert out.loc[1, "NCabin"] == de.DECKS.index("C")   # el mapa de cubiertas es fijo


def test_validate_silver_fails_on_nulls(sample):
    out = de.build_silver(sample)
    out.loc[0, "Fare"] = np.nan
    with pytest.raises(ValueError):
        de.validate_silver(out)


def test_drop_original_columns(sample):
    out = de.drop_original_columns(sample)
    assert not {"Name", "Sex", "Embarked", "Cabin"} & set(out.columns)
    assert "Name" in sample.columns       # la entrada no se modifica