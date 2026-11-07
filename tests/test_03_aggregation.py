import importlib

import pandas as pd
import pytest

agg = importlib.import_module("flows.03_aggregation")


@pytest.fixture
def silver_like():
    return pd.DataFrame({
        "Survived": [1, 1, 0, 0, 1, 0, 0, 1],
        "Pclass": [1, 1, 1, 3, 3, 3, 3, 3],
        "Sex": ["female", "male", "male", "female", "female", "male", "male", "male"],
        "Embarked": ["S", "C", "S", "Q", "S", "S", "C", "Q"],
        "Fare": [80.0, 90.0, 70.0, 7.0, 8.0, 7.5, 9.0, 300.0],
        "Age": [30.0, 40.0, 50.0, 20.0, 25.0, 30.0, 35.0, 28.0],
        "Familia": [0, 1, 0, 2, 0, 0, 1, 0],
        "Alone": ["Alone", "with Family", "Alone", "with Family", "Alone", "Alone",
                  "with Family", "Alone"],
    })


def test_outlier_mask_iqr_flags_only_extreme_values():
    mask = agg.outlier_mask_iqr(pd.Series([1, 2, 3, 4, 5, 100]))
    assert mask.tolist() == [False, False, False, False, False, True]


def test_survival_by_class_sex_is_a_valid_rate_table(silver_like):
    out = agg.survival_by_class_sex(silver_like)
    assert "Pclass" in out.columns
    rates = out.drop(columns="Pclass")
    assert ((rates.dropna() >= 0) & (rates.dropna() <= 1)).all().all()
    assert all(isinstance(c, str) for c in out.columns)      # requisito de Parquet


def test_fare_quartiles_are_computed_within_class(silver_like):
    out = agg.add_fare_quartile(silver_like)
    for _, group in out.groupby("Pclass"):
        assert group["FareQ"].between(0, 3).all()
    assert "FareQ" not in silver_like.columns                 # no muta la entrada


def test_chi2_returns_valid_p_value(silver_like):
    res = agg.chi2_sex_survival(silver_like)
    assert set(res) == {"chi2", "p_value", "dof"}
    assert 0 <= res["p_value"] <= 1


def test_run_writes_gold_tables(tmp_path, silver_like):
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    silver_like.to_parquet(silver_dir / "titanic_silver.parquet", index=False)
    out = agg.run(in_dir=silver_dir, out_dir=tmp_path / "gold")
    assert (out / "survival_by_class_sex.parquet").exists()
    assert (out / "chi2_sex_survival.json").exists()