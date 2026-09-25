"""Unit tests for cases.infra.preflight."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cases.infra._util import sha256_file
from cases.infra.preflight import (budget_check, disk_check, run_preflight_checks,
                                  verify_decision_registry, verify_sidecars)


class RegistryVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_registry(self) -> Path:
        registry = self.root / "preregistry_test_v3.json"
        registry.write_text(json.dumps({"seeds": [0, 1], "caps": {}}))
        digest = sha256_file(registry)
        sidecar = registry.with_suffix(".sha256")
        sidecar.write_text(f"sha256:{digest}\n")
        return registry

    def test_sidecar_pass_and_fail(self) -> None:
        registry = self._write_registry()
        checks = verify_sidecars(self.root)
        self.assertTrue(all(c.ok for c in checks))
        registry.write_text(json.dumps({"seeds": [0, 1, 2], "caps": {}}))
        checks = verify_sidecars(self.root)
        self.assertTrue(any(not c.ok for c in checks))

    def test_decision_registry_pass_and_fail(self) -> None:
        registry = self._write_registry()
        # Repo convention: registry_hash covers the canonical envelope
        # {schema_version, title, created_at, content} (see
        # cases.protocol.registry.canonical_registry_bytes).  Build a document
        # that satisfies it so the check passes wherever cases.protocol is
        # importable, and fails only on pin mismatch.
        document = {
            "schema_version": 3,
            "title": "unit-test registry",
            "created_at": "2026-09-01T00:00:00Z",
            "content": {"seeds": [0, 1], "caps": {}},
        }
        envelope = {key: document[key] for key in
                    ("schema_version", "title", "created_at", "content")}
        digest = hashlib.sha256(json.dumps(
            envelope, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")).hexdigest()
        document["registry_hash"] = f"sha256:{digest}"
        registry.write_text(json.dumps(document))
        decision = self.root / "DECISION.json"
        decision.write_text(json.dumps(
            {"registries": {"causal": {"file": registry.name,
                                       "v3_hash": f"sha256:{digest}"}}}))
        checks = verify_decision_registry(decision)
        self.assertTrue(all(c.ok for c in checks))
        decision.write_text(json.dumps(
            {"registries": {"causal": {"file": registry.name,
                                       "v3_hash": "sha256:" + "0" * 64}}}))
        checks = verify_decision_registry(decision)
        self.assertTrue(any(not c.ok for c in checks))


class DiskAndBudgetTest(unittest.TestCase):
    def test_disk_thresholds(self) -> None:
        with TemporaryDirectory() as tmp:
            ok = disk_check(tmp, threshold_pct=100)
            tripped = disk_check(tmp, threshold_pct=0)
        self.assertTrue(ok.ok)
        self.assertFalse(tripped.ok)

    def test_budget_ceiling_gate_and_drift_warning(self) -> None:
        budget = {"ceiling": 2, "lines": {
            "a": {"pattern": "whatever", "expect": 1},
            "b": {"pattern": "other", "expect": 0},
        }}
        counter = {"whatever": 1, "other": 0}.get
        checks = budget_check(budget, counter=counter)
        self.assertTrue(all(c.ok for c in checks))
        severities = {c.name: c.severity for c in checks}
        self.assertEqual(severities["budget:ceiling"], "fail")

    def test_budget_ceiling_violation_fails(self) -> None:
        budget = {"ceiling": 1, "lines": {"a": {"pattern": "whatever"}}}
        counter = {"whatever": 2, "other": 0}.get
        checks = budget_check(budget, counter=counter)
        gate = [c for c in checks if c.name == "budget:ceiling"][0]
        self.assertFalse(gate.ok)

    def test_composed_checks_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "r.json"
            registry.write_text("{}")
            (root / "r.sha256").write_text(f"sha256:{sha256_file(registry)}\n")
            checks = run_preflight_checks(
                registry_dir=root, disks={str(root): 100})
        self.assertTrue(checks)
        self.assertTrue(all(c.ok for c in checks))


if __name__ == "__main__":
    unittest.main()
