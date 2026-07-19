import json

import pytest

from library_learning.compose import extract


def test_existing_splits_json_wins(tmp_path, monkeypatch):
    splits = {"seed_pids": [0, 1], "composition_validation_pids": [2],
              "reconstruction_pids": [3], "test_pids": [3]}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    monkeypatch.setattr(extract, "make_splits",
                        lambda *a, **k: pytest.fail("make_splits must not run"))
    assert extract.resolve_splits(None, None, tmp_path) == splits


def test_falls_back_to_make_splits(tmp_path, monkeypatch):
    sentinel = {"seed_pids": [9]}
    monkeypatch.setattr(extract, "make_splits", lambda *a, **k: sentinel)
    assert extract.resolve_splits(None, None, tmp_path) is sentinel
