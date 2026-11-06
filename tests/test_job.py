import importlib

from flows import job


def test_stages_are_ordered_and_importable():
    assert list(job.STAGES) == ["ingestion"] #, "silver", "gold", "train"]
    for module_name in job.STAGES.values():
        assert callable(importlib.import_module(module_name).run)


def test_all_runs_stages_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(job, "run_stage", lambda name: calls.append(name))
    job.main(["--stage", "all"])
    assert calls == ["ingestion"] #, "silver", "gold", "train"]