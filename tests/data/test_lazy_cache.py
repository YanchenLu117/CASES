"""Lazy/cached JSON loading contracts (V9 infra)."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cases.data.lazy import LazyDataset, json_cached


@pytest.fixture()
def big_json(tmp_path):
    payload = {
        "metadata": {"nodes": [f"n{i}" for i in range(200)]},
        "tasks": [
            {"id": f"t{i}", "obs": [i, i + 1, i + 2], "attrs": list(range(50))}
            for i in range(4_000)
        ],
    }
    p = tmp_path / "big.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p, payload


def test_cached_equals_direct_parse(big_json):
    p, payload = big_json
    assert json_cached(p) == payload


def test_process_memo_returns_same_object(big_json):
    p, _ = big_json
    assert json_cached(p) is json_cached(p)


def test_cache_invalidated_on_file_change(big_json, monkeypatch, tmp_path):
    p, _ = big_json
    monkeypatch.setenv("CASES_JSON_CACHE_DIR", str(tmp_path / "cache"))
    first = json_cached(p)
    payload2 = {"tasks": [{"id": "x"}]}
    p.write_text(json.dumps(payload2), encoding="utf-8")
    assert json_cached(p) == payload2
    assert json_cached(p) is not first


def test_disable_env_forces_plain_parse(big_json, monkeypatch):
    p, payload = big_json
    monkeypatch.setenv("CASES_JSON_CACHE_DISABLE", "1")
    assert json_cached(p) == payload
    assert json_cached(p) is not json_cached(p)  # no memo when disabled


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        json_cached(tmp_path / "nope.json")


def test_lazy_dataset_defers_until_access(tmp_path):
    calls = []

    def loader(path):
        calls.append(path)
        return {"tasks": [1, 2, 3]}

    lazy = LazyDataset(loader, str(tmp_path / "whatever.json"))
    assert calls == []          # nothing loaded at construction
    assert len(lazy["tasks"]) == 3  # first access triggers load
    assert calls == [str(tmp_path / "whatever.json")]
    assert lazy._loaded == {"tasks": [1, 2, 3]}


def _boom_loader(path):
    raise AssertionError("must not be called during pickling")


def test_lazy_dataset_pickles_without_loading(tmp_path):
    lazy = LazyDataset(_boom_loader, "/nonexistent/deferred.json")
    clone = pickle.loads(pickle.dumps(lazy))
    assert clone._loaded is None
    assert clone._path == "/nonexistent/deferred.json"
