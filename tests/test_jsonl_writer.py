"""B5 writer regression tests: concurrent appends never interleave; no illegal JSON chars."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from jsonl_writer import append_jsonl  # noqa: E402


def test_concurrent_appends_do_not_interleave(tmp_path):
    path = str(tmp_path / "results.jsonl")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: append_jsonl(path, {"i": i, "expr": "x" * 300}), range(200)))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 200
    assert sorted(json.loads(l)["i"] for l in lines) == list(range(200))


def test_illegal_chars_never_written(tmp_path):
    path = str(tmp_path / "results.jsonl")
    append_jsonl(path, {"arm": "native", "task": "Duffing", "temp": 0.2, "r2": None})
    try:
        append_jsonl(path, {"arm": "bad", "r2": float("nan")})
        raised = False
    except ValueError:
        raised = True
    text = open(path, encoding="utf-8").read()
    assert raised, "NaN must be rejected, not serialized into the file"
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text.splitlines()[0])["r2"] is None
