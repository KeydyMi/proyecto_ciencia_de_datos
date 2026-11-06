import importlib

import pandas as pd

ing = importlib.import_module("flows.00_ingestion")


def _fake_source(tmp_path):
    df = pd.DataFrame(
        {"PassengerId": range(1, 6), "Survived": [0, 1, 1, None, None], "Sex": list("mffmm")}
    )
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    return str(path), df


def test_split_labeled():
    df = pd.DataFrame({"Survived": [0, 1, None, None]})
    labeled, unlabeled = ing.split_labeled(df, 2)
    assert len(labeled) == 2 and len(unlabeled) == 2
    assert unlabeled["Survived"].isna().all()


def test_run_writes_bronze_and_metadata(tmp_path):
    source, _ = _fake_source(tmp_path)
    out = tmp_path / "bronze"
    meta = ing.run(source=source, n_labeled=3, out_dir=out)
    assert (out / "titanic_labeled.csv").exists()
    assert (out / "titanic_unlabeled.csv").exists()
    assert (out / "_ingestion_metadata.json").exists()
    assert meta["rows_labeled"] == 3 and meta["rows_unlabeled"] == 2
    assert len(meta["sha256_labeled"]) == 64


def test_bronze_is_an_unmodified_copy(tmp_path):
    source, original = _fake_source(tmp_path)
    out = tmp_path / "bronze"
    ing.run(source=source, n_labeled=3, out_dir=out)
    saved = pd.read_csv(out / "titanic_labeled.csv")
    pd.testing.assert_frame_equal(saved, original.iloc[:3].reset_index(drop=True))